# Changelog

## [Unreleased]

## [0.13.0] — 2026-04-20

Agent-marketplace activation. carl.camp backend endpoints are live
(`POST /api/agents/register` + `POST /api/sync/agent-cards` with
envelope + rate-limit + migration 021 on a2a_agents), so carl-studio
cuts over to the real surface. Production `@coherence_gate` wiring
lands on the publish path — the first concrete call-site adoption.

### Added

- **`carl agent register <name>`** — MIT-clean CLI command. Writes
  locally always; pushes to carl.camp when a bearer token is present
  (env var `CARL_CAMP_TOKEN` or `~/.carl/camp_token`). `--local-only`
  skips the network path; `--org <id>` targets a specific org.
  Flow: `POST /api/agents/register` mints the recipe-shell UUID →
  carl-studio replaces the local placeholder agent_id → calls
  `POST /api/sync/agent-cards` to publish.
- **`carl agent publish`** — pushes all locally-stored cards (or one
  specific via `--agent-id`) to carl.camp. **Coherence-gated via
  `@coherence_gate(min_R=0.5, feature="agent.publish")`**: denies when
  recent success rate is below threshold. Uses
  `success_rate_probe` as the default endogenous probe. First
  production call site for the v0.11 CoherenceGate primitive — Fano V7
  realization now ~90%.
- **`carl agent list [--limit N]`** — enumerate locally-stored cards.
- **`CampSyncClient.register_recipe_shell()`** — Python method
  mirroring the backend `POST /api/agents/register` contract. Returns
  typed `RegisterResult` with `{agent_id, org_id, lifecycle_state,
  created_at}` or structured error.
- **`SyncResult.envelope_ok`** — captures the backend's
  `{ok: true/false, ...}` envelope so call sites can distinguish
  transport success from business-logic success.

### Changed

- `AgentCardStore._conn_ctx()` normalizes both LocalDB connection
  shapes (context-manager + raw). Enables use from both carl-studio's
  real LocalDB and test-helper wrappers.

### Verification

- Tests: 3070 pass / 0 fail (+10 marketplace-flow covering
  register_recipe_shell happy/4xx/429/transport, envelope handling,
  CLI local-only + missing-token paths, content_hash required).
- Build: 0.13.0 wheel clean.
- Backend integration verified against contract shape (envelope +
  error codes + rate-limit headers) via mocked transport; live-
  endpoint smoke requires `CARL_CAMP_TOKEN`.

### Deferred (still on the v0.14 list)

- Tool-dispatch extraction from `chat_agent.py` (blocked on
  `tool_dispatcher.py` API extension; non-trivial).
- `carl env` expanded to full 7-question design (reward · cascade ·
  eval questions).

## [0.12.0] — 2026-04-20

Decomposition + wizard release. First cut of the long-deferred
chat_agent.py god-class decomposition, plus the `carl env`
progressive-disclosure wizard MVP.

### Added

- **`carl env`** — new top-level CLI command. 4-question wizard
  (mode · method · dataset · compute) that builds a `carl.yaml`
  training config. Resume-capable via `~/.carl/last_env_state.json`.
  Flags: `--resume`, `--auto`, `--json`, `--dry-run`, `--output`.
  Functor-composed questions so answer order doesn't matter when
  fields are disjoint.
- **`src/carl_studio/env_setup/`** new package — `state.py`
  (`EnvState` Pydantic model), `questions.py` (registry +
  `next_question`), `render.py` (YAML emission).

### Changed

- **`SessionStore` extracted** from `chat_agent.py` to a new
  `src/carl_studio/sessions.py` module. First cut of the
  multi-session god-class decomposition (was 3x deferred → auto-P1
  per Anti-Deferral Protocol). `chat_agent.py` re-imports the
  extracted names for back-compat; all existing callers continue to
  work unchanged. `chat_agent.py` shrinks ~170 LOC.

### Deferred (remaining god-class decomp scope)

- Tool-dispatch loop extraction (`chat_agent.py:1280-1475`) →
  candidate for v0.13 once `tool_dispatcher.py` gains the needed
  API. Coherence probe lives here.
- One-shot inference path (`_one_shot_text`, `_build_system_prompt`)
  → v0.13.
- Remaining `CARLAgent` class (~1700 LOC) → expected to settle
  naturally as tool-dispatch and prompt-building extract.

### Verification

- Tests: 3060 pass / 0 fail (+19 carl-env; 3041 → 3060). All v0.11
  surfaces unchanged. SessionStore tests pass via both the new
  import path (`carl_studio.sessions`) and the legacy path
  (`carl_studio.chat_agent`).
- Build: 0.12.0 wheel clean.

## [0.11.0] — 2026-04-20

Fano-followthrough release. Closes the two P1-P2 findings that v0.10.0
left open + ships the first v0.9-designed feature (`carl update`).

### Added

- **Step.probe_call audit trail** (Fano V5 witnessability). When a
  registered coherence probe populates phi/kuramoto_r/channel_coherence,
  the Step records `{probe_name, inputs_sha256, output_sha256, populated}`
  — 12-hex digests, not full payloads, to preserve BITC axiom 1 bounded
  support. Serialized via `Step.to_dict()`.
- **`success_rate_probe`** in `carl_core.presence`. A default endogenous
  probe: reads the chain's own tail of same-action steps and returns
  `{kuramoto_r: success_rate}`. Pairs with `@coherence_gate` to close
  the IRE "G" realization end-to-end (Fano V7 45% → ~75%). Exported
  from `carl_core.__init__`.
- **`carl update` command** + `carl_studio.update` package. Surfaces
  recent git commits, PyPI dep-version deltas, and positive-framed
  blast-radius summary. `--dry-run` skips network; `--json` emits
  machine-readable; `--summary-only` for one-liner; `--detailed` for
  full lists. Consent-gated for network egress.

### Changed

- `Step` schema gained optional `probe_call` field (additive,
  backward-compatible).
- `carl_core.__init__` exports `success_rate_probe` alongside existing
  presence helpers.

### Verification

- Tests: 3041 pass / 0 fail (3026 → 3041, +15 for `carl update`).
- Zero feature regression. All v0.10 surfaces unchanged.
- Build: 0.11.0 wheel + sdist clean.

## [0.10.0] — 2026-04-20

Architecture-completion release. Closes the four gaps the four-agent
vanilla peer review flagged against v0.8.0, plus shipping the initial
marketplace agent-card client, the coherence-gated routing primitive,
and the presence-report query helper. Validated by a Fano-plane (K_7)
consensus pass across seven axes: boundedness, recurrence, endogenous
measurability, contrastive coherence, witnessability, manifold
integrity, gate realization.

v0.9 was skipped as a release tag — all v0.9-design work
(``carl-update``, ``carl-env``) ships in v0.10 alongside the v0.10-A
primitives.

### Added

- **CoherenceGate primitive** (``carl_studio.gating``). Closes the
  ``G`` in IRE's ``(M, I, Φ, G)`` tuple. ``CoherenceGatePredicate``
  reads tail-window Kuramoto R from the active chain; ``@coherence_gate(min_R=...)``
  decorator raises ``CoherenceError(code="carl.gate.coherence_insufficient")``
  when R is below threshold. Opt-in — stacks with ``consent_gate`` /
  ``tier_gate``. ``CoherenceSnapshot.is_degenerate`` + ``variance``
  field flag constant-probe signals without forcing deny.
- **Coherence auto-attach on InteractionChain.record()**
  (``carl_core.interaction``). Opt-in
  ``register_coherence_probe(fn)`` callback invoked at record time for
  ``LLM_REPLY`` / ``TOOL_CALL`` / ``TRAINING_STEP`` / ``EVAL_PHASE`` /
  ``REWARD`` action types when no explicit coherence kwargs are
  passed. Probe exceptions swallowed; non-dict returns ignored.
  Explicit kwargs always override the probe.
- **PresenceReport + compose_presence_report** (``carl_core.presence``).
  Thin composition helper — NOT a new primitive. Returns a frozen
  dataclass with R, psi, crystallization, Deutsch-Marletto
  ``constructive`` flag, recent action types, and a human-readable
  note. Registered as MCP tool ``carl.presence.self`` for agent
  self-introspection.
- **Marketplace agent cards** (``carl_studio.a2a.marketplace``).
  ``MarketplaceAgentCard`` Pydantic model aligned with the carl.camp
  ``POST /api/sync/agent-cards`` contract; ``AgentCardStore`` local
  SQLite persistence with paginated ``list_all(limit, offset)``;
  ``CampSyncClient`` HTTP push with pluggable transport + 429 +
  batch-limit handling. ``content_hash`` canonicalization
  (sha256 over sorted-keys JSON). Distinct from the existing
  ``CARLAgentCard`` (running-instance manifest).
- **Tool-call witness completeness** (``chat_agent.py``). Every tool
  dispatch — ok, denied, schema_error, error — records an
  ``ActionType.TOOL_CALL`` step on the InteractionChain with
  ``{outcome, result}`` payload and measured ``duration_ms``. Closes
  the pre-v0.10 gap where CLI + memory were logged but tool calls
  were not. Fire-and-forget recording; chain persistence failures
  never propagate.
- **packages/carl-core/LICENSE** — MIT text mirrored from repo root.
  carl-core ships as a separate wheel and now carries its own
  license file.

### Changed

- **``emit_gate_event``** extended with optional ``gate_code``
  parameter that surfaces in the step output dict for downstream
  filtering. Back-compat: default ``None`` preserves v0.8 behavior
  for existing callers.
- **``docs/private_integration.md``** now documents the
  ``load_private()`` three-layer fallback contract (hardware-HMAC
  → ``terminals-runtime`` → HF private dataset → MIT-safe stub) +
  non-obligations (no pre-check required, no caching required).

### Added — governance

- **Fano-plane peer-review pattern** (``AGENTS.md``). Dispatching
  7 vanilla-context agents aligned to BITC/IRE axes (N=7 = K_7,
  complete mutual observation per BITC §6.1) before any major
  release tag. Each writes JSON-DAG findings; MECE coalesce
  produces consensus. Anti-patterns flagged directly feed
  ``CLAUDE.md`` for future-session filtering.

### Deferred to v0.11

- Step schema extension for probe audit trail (``step.probe_call``
  sub-field) — Fano V5 witnessability finding.
- Typed context manifold on InteractionChain — Fano V6 forward.
- Applying ``@coherence_gate`` to production call sites (training
  admission, marketplace publish, etc.) — Fano V7 flagged zero
  production call sites today. The primitive is demonstrated
  end-to-end via ``tests/test_fano_consensus_fixes.py`` but live
  wiring is explicit v0.11 scope.
- ``chat_agent.py`` further decomposition (2,443 LOC) — auto-promotes
  to P1 if re-deferred per Anti-Deferral Protocol.

### Verification

- **Tests:** 3009 pass / 0 fail (2923 v0.8 core → 3009 now, +86 new).
- **Peer review:** two waves (4-agent v0.10 review + 7-agent Fano
  consensus K_7). All findings addressed or explicitly deferred with
  rationale.
- **Build:** ``python -m build`` produces clean 0.10.0 wheel + sdist.
- **IP boundary:** MIT carl-studio unchanged; no BUSL methodology
  copied; admin-gate + lazy-import seam preserved.
- **κ:** ``KAPPA = 64 / 3`` unchanged per Tej's ruling
  (exact from early Desai papers; terminals.tech's 21.37 is
  downstream calibration).

## [0.8.0] — 2026-04-20

Consolidation release. No new product surfaces — four crystallization tracks
collapse duplicated patterns from the v0.5→v0.7.1 arc into typed primitives,
expose named plug-points for private-runtime extension, and publish the
follow-up paper series justified by shipped work. Grounded in a four-agent
review (isomorphism map · IP boundary · paper series · integration seams).

### Added

- **`BaseGate[P: GatingPredicate]`** in `carl_studio.gating` — shared
  generic owning the predicate → emit → raise loop. `consent_gate` and
  `tier_gate` delegate to it internally; public signatures, error codes,
  and decorator shapes are unchanged.
- **`ConfigRegistry[T: BaseModel]`** in `carl_studio.config_registry` —
  typed wrapper over `LocalDB.get_config/set_config` with Pydantic v2
  validation. Auto-derived `namespace.modelname` keys; schema mismatch
  raises `CARLError(code="carl.config.schema_mismatch")`. `LocalDB`
  gained `.config_registry(cls, *, namespace, key=None)` factory.
  `SpendTracker` migrated to persist `SpendState` under
  `carl.x402.spendstate` — legacy two-key format is auto-migrated on
  first read (opt-out via `CARL_CONFIG_MIGRATE=skip`).
- **`BreakAndRetryStrategy`** in `carl_core.resilience` — composes
  `RetryPolicy` and `CircuitBreaker` behind one `.run()` / `.run_async()`
  call. Raises `CircuitOpenError(code="carl.resilience.circuit_open")`
  when the breaker is open. x402 facilitator calls gained a strategy
  binding alongside the existing breaker (additive).
- **`carl_studio.x402.register_confirm_callback(name, cb)`** — named
  registry so `X402Config.confirm_payment_cb: str | Callable | None`
  can be resolved at execute time. Private runtimes persist a callback
  name via carl.camp settings; direct `Callable` path unchanged.
- **`carl_studio.metrics.public_registry()`** +
  `register_external_collector(collector)` — shared `CollectorRegistry`
  accessible to private dashboards; external collectors surface on
  `carl metrics serve` automatically (same registry).
- **`carl_studio.tier.register_tier_resolver(fn)`** — pluggable tier
  source. `TierPredicate._effective()` checks the resolver before
  falling back to `detect_effective_tier()`. Errors wrap as
  `CARLError(code="carl.tier.resolver_error")`.
- **Paper series.** `paper/carl-paper.md` → `paper/01-main-carl.md`.
  Added `02-phase-adaptive-methods.md`, `03-coherence-trap-technical-note.md`,
  `04-interaction-chains-witness-logs.md`, plus `docs/paper_series.md`
  index. All cross-references verified against v0.7.1 symbols.
- **`docs/private_integration.md`** — examples for the three plug-points.

### Changed

- **`consent_gate` / `tier_gate` internal shape.** Both now thin
  delegates over `BaseGate`. External contract (signatures, error
  classes, error codes, decorator metadata) unchanged. 18 new
  `tests/test_gating_base.py` tests pin the shared primitive.
- **`SpendTracker` persistence format.** Legacy `carl.x402.spend_today` /
  `carl.x402.daily_reset_at` keys are replaced by `SpendState` JSON at
  `carl.x402.spendstate`. Idempotent migration on first read.
- **`X402Config.confirm_payment_cb` type.** Widened to
  `str | ConfirmPaymentCallback | None`. Existing Callable users see no
  behavior change.
- **`PhaseTransitionGate`** moved from inline in `carl_studio/__init__.py`
  to `carl_studio/training/gates.py`. Re-exported via lazy
  `__getattr__` — import time stays under 200ms. Seed-first
  resolution preserved.

### Removed

- `_ConsentFlagKeyShim` and module `__getattr__` deprecation trampoline
  in `consent.py` (marked for v0.8 removal since v0.6.3). Call sites
  must use the `ConsentKey` Literal directly.
- `TierError = TierGateError` alias in `carl_studio/agent/tier_gate.py`.
  Import `TierGateError` from the canonical location.
- Stale filesystem artifacts: `src/carl_studio/cli.py` (pre-CLI-collapse
  monolith), `src/carl_studio/x402_sdk.py` (moved to `x402_connection.py`
  in v0.5.0), `src/carl_studio/primitives/` (removed in v0.5.0 but
  Finder-restored as cruft). `tests/test_x402_sdk.py::test_x402_sdk_module_gone`
  now passes.

### Papers

See `docs/paper_series.md` for the full index and cross-reference table
(paper ↔ shipped code path). Four-paper series covering main framework,
phase-adaptive methods, the coherence trap, and interaction-chain
witness logs.

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
