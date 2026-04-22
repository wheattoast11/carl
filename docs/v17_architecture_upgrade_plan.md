---
last_updated: 2026-04-21
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.17.0-planned
classification: internal — v0.17 execution blueprint
---

# v0.17 Architecture Upgrade Plan

**MBB-style execution blueprint. Foundation sprint + three parallel post-foundation tracks + one cross-cutting policy track. Every workstream produces a typed deliverable. Every team lead signs off with a structured readout.**

---

## 0. Executive summary

**Six teams**, one foundation sprint running parallel with a **moat extraction sprint**, three parallel post-foundation tracks, one cross-cutting policy track.

**Outcome.** A unified handle-runtime primitive (`Vault[H, V]` + resolver chain), a `Session` top-level entry point, a canonical `TrainingSpec`, `chat_agent.py` split into a package, `@audited` + `InteractionChain` contextvar, content-addressed audit steps, an explicit public/private moat seam — **and extraction of ~1,500 LOC of novel IP currently MIT-licensed in the public repo to private / `terminals-runtime` where it belongs.**

**Moat leak finding (critical).** The UX/IP audit found that:

| File | LOC | What's leaking | Disposition |
|---|---:|---|---|
| `packages/carl-core/src/carl_core/constitutional.py` | 556 | 25-dim action feature encoder, ed25519-signed hash-chained ledger, genesis signing-bytes layout | Extract to `terminals-runtime/constitutional/`; leave thin `ConstitutionalLedgerClient` in carl-core |
| `packages/carl-core/src/carl_core/heartbeat.py` | 436 | Standing Wave Theorem docstring + coefficient-pinned reference implementation | Keep API + theorem docs public; move pinned coefficients + reference impl private |
| `packages/carl-core/src/carl_core/resonant.py` (`joint`-mode cognize path) | ~80 of 504 | `joint` cognition collapses latent vector → scalar readout; v0.16 commercial differentiator | Extract the `joint` math; keep `per_dim` public |
| `src/carl_studio/training/rewards/eml.py` (trained weights + init heuristics) | ~150 of 427 | 7-param tree trained at +0.972 correlation with PhaseAdaptive — the benchmark-validated numbers | Move weights + init to `terminals-runtime/rewards/`; keep structure public |

**Total: ~1,220 LOC of IP moves from MIT carl-studio/carl-core to private repos.** Now is the moment — before any published consumers fork the public version.

**Delta.** +~1,400 LOC added (new primitives), −~2,000 LOC removed (unified vault), −~1,220 LOC extracted to private repos, net −~1,820 LOC on public surface. ~60 files changed across 6 workstreams. 1 new PAID tier key (`secrets.managed_vault`). No FREE tier reductions for existing users. Zero public wire-format breaks (EML protocol, npm codec parity, κ = 64/3, paper references preserved).

**Timeline.** Two weeks calendar with parallel execution; ~50 hours focused engineering. Foundation + Moat Extraction run parallel as the critical path (4 days). Tracks B/C/D/E run concurrently after those merge.

**Posture.** No backward compatibility obligation (pre-launch). We preserve external wire formats + user-typed schemas + mathematical constants; we break internal APIs freely and we relocate IP aggressively.

**Live gap (flagged during audit).** `src/carl_studio/chat_agent.py` does NOT use the `HandleRuntimeBundle` I shipped in v0.16.1 — the main consumer bypasses the canonical wiring. Team C's chat_agent split MUST adopt the bundle during the refactor.

---

## 1. Context + constraints

### 1.1 Why now

1. Pre-launch — no external users; internal APIs are fair game.
2. v0.16.1 just landed the handle runtime; three vaults have isomorphic shape now ripe for unification.
3. `chat_agent.py` at 2,281 LOC is hitting operational limits (debugging friction, test runtime per-change).
4. Terminals OS convergence (mid-to-late 2026 target) requires the cleanest primitive shape in `carl-core` that can migrate wholesale into `terminals-core`.
5. The digital-twin positioning demands `Session.snapshot()` / `restore()` — which requires the vault + chain unification as a prerequisite.

### 1.2 What must NOT break

| Contract | Source of truth | Why it matters |
|---|---|---|
| EML wire format | `docs/eml_signing_protocol.md` + `packages/emlt-codec-ts/test/vectors.json` | Shared with `@terminals-tech/emlt-codec` npm sibling |
| κ = 64/3 | `packages/carl-core/src/carl_core/constants.py` | Zenodo DOIs reference the exact value |
| `carl.yaml` top-level schema | User-typed; paper references | Breaking this breaks published docs |
| `carl.<namespace>` error codes | Convention | Downstream tooling branches on codes |
| Python ≥ 3.11 floor | `pyproject.toml` | Committed lower bound |

### 1.3 What we ARE allowed to break

- All internal Python module paths.
- `SecretVault` / `DataVault` / `ResourceVault` class shapes (collapse to `Vault[H, V]`).
- `HandleRuntimeBundle.build()` surface (subsumed by `Session`).
- Adapter `translate_config()` signatures (unified under `Spec → translate()`).
- `chat_agent.py` file path (package split).
- Any test private-method hook (tests re-baseline).

### 1.4 Data-backed blast radius (measured 2026-04-21)

| Change | Files | Occurrences | Risk |
|---|---:|---:|---|
| Vault[H,V] unification | ~50 | ~340 | Medium — mechanical |
| Session (additive) | 3 new; opt-in adoption | ~50 callers (long tail) | Low |
| TrainingSpec | 8 | 40 | Low-Medium |
| `chat_agent` split | 14 | 29 | Low — import path renames |
| `@audited` (opportunistic) | 12 | 61 | Low — gradual |

### 1.5 God-class register (flagged, out of scope this plan)

| File | LOC | Disposition |
|---|---:|---|
| `src/carl_studio/chat_agent.py` | 2,281 | **Split in this plan** |
| `src/carl_studio/eval/runner.py` | 2,028 | Flagged for v0.18 |
| `src/carl_studio/training/trainer.py` | 1,485 | Flagged for v0.18 |
| `src/carl_studio/align/pipeline.py` | 1,233 | Flagged for v0.18 |
| `src/carl_studio/a2a/connection.py` | 1,083 | Flagged for v0.18 |
| `src/carl_studio/cli/training.py` | 1,049 | Flagged for v0.18 |
| `src/carl_studio/mcp/server.py` | 986 | Monitor; split later if ≥1,200 |

### 1.6 Out of scope (explicit declines — see §3.9)

- Async-first HTTP rewrite (defer v0.18)
- EML as universal learnable primitive (defer — no consumer)
- Typed FeatureRegistry (flat dict fine until ≥500 features)
- Test folder reorg (cosmetic)
- Durable vault daemon (Fernet-file resolver covers the need)

---

## 2. Data-backed review: which proposals survive scrutiny

Every original proposal is re-validated against the blast-radius data.

### 2.1 ✅ PROPOSAL 1: `Vault[H, V]` + resolver chain

**Original claim.** Three vaults duplicate UUID KV + RLock + `_Entry` slots + TTL + resolve/revoke/exists/list. ~300 LOC copy-paste, three places to miss a lifecycle step.

**Data check.** Confirmed. 50 files, 340 occurrences across the six symbols. Uniform shape across all three.

**Still recommend.** YES.

**Refinement.** Frame as `carl_core.vault.Vault[H, V]` generic base (`H` = handle type, `V` = backend value type). Three specializations become ≤40 LOC each. Resolver chain is a first-class feature of the base, not a secrets-only addition — so data vault's `query` / `url` kinds become functional, resource vault can register MCP-endpoint resolvers, etc. Public resolvers: `env`, `keyring`, `fernet-file`. Proprietary resolvers register via admin-gate (see §2.8).

### 2.2 ✅ PROPOSAL 2: `Session` as top-level — REVISED SCOPE

**Original claim.** `carl.Session(user=...)` context manager bundles everything.

**Data check.** `InteractionChain()` appears 237 times across 48 files — mostly tests. A forced migration is prohibitive.

**Still recommend.** YES — but **ADDITIVE, not replacing**. Tests keep direct `InteractionChain()`. Session is the canonical entry point for app-level code. `Session.chain` exposes the underlying chain for anyone who needs it.

**New insight — Twin Checkpoint.** Session is the right primitive for digital-twin checkpoint/restore. `Session.snapshot() → TwinCheckpoint` / `Session.restore(TwinCheckpoint)` serialize chain + vault refs (values stay behind resolvers) + active resource refs + agent memory. Digital-twin story requires this. +200 LOC.

### 2.3 ✅ PROPOSAL 3: `chat_agent` split

**Original claim.** Split into a package.

**Data check.** 14 files / 29 occurrences — much smaller blast than feared.

**Still recommend.** YES.

**Scope.** Split `chat_agent.py` (2,281 LOC) into `chat_agent/` package:

| Submodule | LOC target | Responsibility |
|---|---:|---|
| `agent.py` | ≤500 | Public `CARLAgent` class + streaming tool-loop |
| `knowledge.py` | ≤400 | Knowledge store (embedding + retrieval + ingest) |
| `memory_interface.py` | ≤300 | Agent-side wrapper over `MemoryStore` |
| `constitution_runtime.py` | ≤300 | Policy evaluation + ledger append |
| `one_shot.py` | ≤300 | `_one_shot_text` / inference-without-loop |
| `prompt.py` | ≤400 | System-prompt builder |

Public import stays `from carl_studio.chat_agent import CARLAgent` via package `__init__.py` re-export.

### 2.4 ✅ PROPOSAL 4: Canonical `TrainingSpec`

**Original claim.** Adapters become pure translators over a shared spec.

**Data check.** 8 files use `translate_config`; no shared Translator protocol exists today.

**Still recommend.** YES.

**Scope.** `carl_core.training.Spec` Pydantic v2 model:

```python
class Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: ModelSpec
    dataset: DatasetSpec
    algorithm: Literal["sft", "grpo", "dpo", "distill"]
    hyperparams: HyperparamSpec
    lifecycle: LifecycleSpec   # epochs, checkpoints, eval gates
    compute: ComputeSpec       # backend, gpus, memory
    rewards: list[RewardSpec] | None = None  # for GRPO
```

Every adapter (trl, slime, unsloth, axolotl, tinker, atropos) migrates to `translate(spec: Spec) -> AdapterNativeArgs`. `carl.yaml` still parses as-is (yaml schema unchanged) — the parse result produces a Spec.

### 2.5 ✅ PROPOSAL 5: `@audited` decorator + `InteractionChain` contextvar

**Original claim.** Decorator-based audit emission.

**Data check.** 12 files, 61 imperative `chain.record(ActionType.X, ...)` call sites.

**Still recommend.** YES. Migration is opportunistic — no bulk refactor.

**Scope.**

```python
@audited(ActionType.TRAINING_STEP, name_fn=lambda self, *a, **k: f"step.{self.run_id}")
def train_step(self, batch: dict) -> dict: ...
```

Decorator pulls the current `InteractionChain` from a contextvar, records before+after, captures duration + success, attaches `probe_call`. Ship the machinery; migrate 3-5 high-value sites in this release (training/pipeline, eval/runner entry, slime_bridge, chat_agent tool loop post-split).

### 2.6 ✅ NEW — PROPOSAL 6: Content-addressed chain steps

**Why.** The audit trail's value depends on immutability. Today `chain.to_jsonl()` is append-only in practice but not cryptographically verifiable.

**Scope.** Compute `content_hash(step)` at serialize time. Emit as `step.content_hash` in the JSONL. Ship one property-based test asserting no scrubbed secret pattern survives in the final JSONL. ~30 LOC. Future terminals-OS kernel can verify chain integrity trivially.

### 2.7 ✅ NEW — PROPOSAL 7: Twin Checkpoint primitive

**Why.** Digital-twin positioning requires explicit snapshot/restore. Currently `InteractionStore`, `MemoryStore`, `sessions.py`, and vaults are all separate persistence concerns.

**Scope.** `Session.snapshot() -> TwinCheckpoint` / `Session.restore(TwinCheckpoint)`. Content-addressed (extends Proposal 6). Does NOT serialize values from vaults — serializes refs only. Resolvers reattach on restore via their own backends (1Password CLI, keyring, Fernet file).

### 2.8 ✅ NEW — PROPOSAL 8: Private-repo moat seam

**Why.** The `/resonance/*` + `terminals-runtime/*` split is the IP moat. Public carl-studio (MIT) must have a clean seam for private extensions to register at runtime without ever being imported at module-load time.

**Scope.** Formalize the admin-gate pattern:

```python
# PUBLIC — in carl-studio
class Vault[H, V]:
    def register_runtime_resolver(
        self, admin_token: AdminToken, kind: Kind, resolver: Resolver,
    ) -> None: ...

# PRIVATE — in terminals-runtime, registered at admin-gate entry
def install_hardware_resolvers(vault: Vault, token: AdminToken) -> None:
    vault.register_runtime_resolver(token, "hardware-attested", HWResolver())
```

Already partly present in `src/carl_studio/admin.py`. Formalize, document, test.

### 2.9 ❌ Declines (with rationale)

| Proposal | Decline reason |
|---|---|
| EML as universal learnable primitive | No motivating consumer; deferring forces a real requirement |
| Async-first HTTP rewrite | Defer until swarm workloads actually land (v0.18+) |
| Typed FeatureRegistry | Flat dict fine until ≥500 features |
| Test folder reorg | Cosmetic; not worth import churn |
| Durable vault daemon | Fernet-file resolver covers the need |

---

## 3. Team structure (MBB model)

Five teams. Team leads are convergence points: they own the workstream spec, sign off on atomic-task completion, and produce the workstream readout. A single integration engineer (me — "Orchestrator") owns cross-team merge and the executive readout.

### Team A — Foundation (Lead: "Vault Architect")
Critical-path sprint. Everything else depends on A's output.

### Team B — Training (Lead: "Training Architect")
Runs parallel after Foundation merges. Consumes `Vault[H, V]` for artifact refs + `Session` for audit context.

### Team C — Agent (Lead: "Agent Architect")
Runs parallel after Foundation. Consumes `Session` for chain ownership + tool-registry binding.

### Team D — Observability (Lead: "Observability Architect")
Runs parallel after Foundation. Consumes `InteractionChain` contextvar from Foundation. Migrates 3-5 high-value audit sites; ships the decorator machinery.

### Team E — Policy / IP (Lead: "Policy Architect")
Runs **parallel to everything** (no foundation dependency). Produces tier updates, moat audit, docs, process policy.

### Team F — Moat Extraction (Lead: "Moat Architect")
Runs **parallel to Team A** (no tech dep — moves files + reshapes admin-gate pattern). Extracts ~1,220 LOC of novel IP from MIT public surface into private repos. Critical: must ship in v0.17 before any public version becomes canonical reference.

---

## 4. Workstreams + atomic tasks

Each task has:
- **Scope** — what's done
- **Output** — typed deliverable (file path + shape)
- **Verification** — how we know it's right
- **Owner** — which team

### 4.A Foundation Sprint (Team A)

**A1. Generic `Vault[H, V]` base class**
- Scope: `packages/carl-core/src/carl_core/vault.py` — `Vault` generic over handle type `H` (bound to `HandleRef` protocol) and backend value type `V`. Methods: `put`, `resolve`, `revoke`, `exists`, `list_refs`, `fingerprint_of`, `__len__`. Thread-safe (RLock). TTL self-revoke at resolve time.
- Output: `vault.py` (~250 LOC). `HandleRef` Protocol in `carl_core.handles`.
- Verification: 40+ unit tests exercising generic invariants. Property-based test (hypothesis) for put→revoke→resolve = error.

**A2. Resolver chain primitive**
- Scope: `Vault.resolvers: dict[Kind, Resolver]`. Registration via `vault.register_resolver(kind, fn)`. `vault.resolve(ref)` falls through local → resolver → `not_found`. TTL cache per-ref.
- Output: extended `vault.py`. `Resolver` Protocol. New error codes: `carl.vault.resolver_not_found`, `carl.vault.resolver_unavailable`.
- Verification: Integration test with a fake resolver that returns deterministic bytes. Cache-hit test. Circuit-breaker test (resolver that raises N times).

**A3. `SecretVault` specialization**
- Scope: `carl_core.secrets.SecretVault` rewritten as `Vault[SecretRef, bytes]` subclass. `resolve(privileged=True)` guard preserved. `SecretRef` shape unchanged (stable wire for logging).
- Output: `secrets.py` reduced from ~480 LOC → ~120 LOC. Error codes unchanged.
- Verification: all 34 existing `test_secrets.py` tests pass.

**A4. `DataVault` specialization**
- Scope: `carl_core.data_handles.DataVault` rewritten as `Vault[DataRef, DataPayload]` where `DataPayload` is a union of bytes/file-path/iterator/external-ref. `put_bytes` / `open_file` / `open_stream` / `put_external` all route through the generic base.
- Output: `data_handles.py` reduced from ~485 LOC → ~180 LOC.
- Verification: all 21 `test_data_handles.py` tests pass.

**A5. `ResourceVault` specialization**
- Scope: `carl_core.resource_handles.ResourceVault` as `Vault[ResourceRef, Any]` with caller-supplied closer-on-revoke.
- Output: `resource_handles.py` reduced from ~250 LOC → ~100 LOC.
- Verification: all 11 `test_resource_handles.py` tests pass.

**A6. Public resolvers**
- Scope: Three ship-ready resolver modules in `src/carl_studio/handles/resolvers/`:
  - `env.py` — `env://VAR` → `os.environb[VAR]`
  - `keyring.py` — `keyring://service/account` → OS keychain (refactor existing `KeychainBackend`)
  - `fernet_file.py` — `fernet-file://~/.carl/vault/<name>` → Fernet-encrypted local file
- Output: 3 files, ~100 LOC each. Integration with `SecretVault.register_resolver`.
- Verification: 12 integration tests (4 per resolver). End-to-end: mint → resolve → type-from-secret in browser test.

**A7. `Session` primitive + checkpoint**
- Scope: `src/carl_studio/session.py`. Context manager. Owns chain + secret_vault + data_vault + resource_vault + subprocess_toolkit + browser_toolkit + cu_dispatcher. `Session.snapshot() → TwinCheckpoint` / `Session.restore()`.
- Output: `session.py` (~300 LOC). `TwinCheckpoint` Pydantic model.
- Verification: 20 tests. Snapshot round-trip test. `with carl.Session() as s:` works for 3 example flows (agent-only, browser, subprocess).

**A8. Admin-gate resolver registration seam**
- Scope: Formalize `Vault.register_runtime_resolver(admin_token, kind, resolver)` — private-repo hook point. Document in `docs/v17_vault_resolver_chain.md`.
- Output: `AdminToken` type in `carl_studio.admin`, `register_runtime_resolver` method on `Vault` base.
- Verification: test that unauthorized token raises `carl.admin.unauthorized`. Mock hardware resolver to confirm the seam works end-to-end.

**A9. Team A readout**
- Output: `docs/v17_team_a_readout.md` — what shipped, full-suite result, blast radius actual vs predicted, any scope cuts.
- Sign-off criteria: full regression green, zero pyright strict errors on new surfaces, zero ruff warnings.

---

### 4.B Training (Team B)

**B1. `carl_core.training.Spec`**
- Output: `packages/carl-core/src/carl_core/training/spec.py` (~250 LOC). Pydantic v2, `frozen=True`, `extra="forbid"`.
- Verification: JSON-schema export + round-trip from `carl.yaml` fixtures.

**B2-B7. Adapter migrations** — TRL, Slime, Unsloth, Axolotl, Tinker, Atropos
- Each: `translate(spec: Spec) → AdapterNativeArgs`. No behavior change.
- Output: one module per adapter updated. `SlimeArgs.json_schema()` output validated against existing carl.camp expectations.
- Verification: every adapter's existing test suite passes unchanged. Golden-file test: `carl.yaml` → Spec → translate → expected argv.

**B8. `carl.yaml` parser**
- Output: `parse_carl_yaml(path) → Spec`. Lives in carl-studio (filesystem touch).
- Verification: 10 fixture yamls parse successfully. Malformed yaml produces `carl.training.spec_invalid` with named field + line number.

**B9. Team B readout**
- Same shape as A9.

---

### 4.C Agent (Team C)

**C1–C6. Submodule extractions** per §2.3 table.
- Each submodule ships with its own tests.
- Verification per submodule: LOC ≤ target; public API unchanged.

**C7. Package `__init__.py` re-export**
- Output: `src/carl_studio/chat_agent/__init__.py` re-exports `CARLAgent`, `ToolPermission`, etc.
- Verification: `from carl_studio.chat_agent import CARLAgent` works unchanged; all 85+ chat_agent tests pass.

**C8. Session integration**
- Output: `CARLAgent.__init__` accepts an optional `Session` parameter; if present, uses `session.chain` / `session.tool_dispatcher`.
- Verification: new `test_agent_with_session.py` — 8 tests exercising the Session-bound agent flow.

**C9. Team C readout**
- Same shape.

---

### 4.D Observability (Team D)

**D1. `@audited` decorator + contextvar**
- Output: `carl_core.audit` module. `@audited(ActionType, name_fn=...)` decorator. `InteractionChainContext` contextvar.
- Verification: decorator + contextvar unit tests. Failure-mode test: exception raised in decorated method produces a step with `success=False` and `duration_ms`.

**D2. Migrate 5 high-value sites**
- Scope: `training/pipeline.py::train_step`, `eval/runner.py::run_eval_phase`, `slime_bridge.py::score_completion`, `chat_agent/agent.py::tool_loop_iteration` (post-split), `cu/browser.py::screenshot` (already records; validate contextvar).
- Verification: chain output matches byte-for-byte before/after migration for deterministic test cases.

**D3. Content-addressed steps** (Proposal 6)
- Scope: `Step.content_hash` field computed at `to_dict()` time. Hash inputs: canonical JSON of step minus the hash field itself. Property test: re-serializing produces the same hash.
- Output: extension to `carl_core.interaction.Step`. ~30 LOC.
- Verification: property-based test via hypothesis. Scrubber-integration test: asserts no secret pattern survives in the hashed output.

**D4. Unified scrubber registry** (stretch, drop if time-constrained)
- Scope: Consolidate `interaction._SECRET_PATTERNS` + `cu/privacy.py` into one `carl_core.scrubber` module with a `register_pattern(category, pattern)` API.
- Verification: all existing secret + PII test vectors pass through the new registry.

**D5. Team D readout**

---

### 4.F Moat Extraction (Team F)

**F1. Constitutional ledger extraction**
- Scope: Move `packages/carl-core/src/carl_core/constitutional.py` (556 LOC) → `resonance/src/resonance/signals/constitutional.py`. Keep a thin `ConstitutionalLedgerClient` in carl-core that communicates via the admin-gate pattern: on `load_private("signals.constitutional")`, the client binds to the private runtime's ledger; without private runtime, methods raise `carl.constitutional.private_required`.
- Output: ~556 LOC moved to resonance; ~80 LOC `ConstitutionalLedgerClient` added to carl-core. Error code `carl.constitutional.private_required`.
- Verification: ledger genesis + append + verify round-trip still works when admin-gate resolves; fails cleanly without. All 6 existing `test_constitutional.py` tests migrate to test the client + the private-runtime stub.

**F2. Heartbeat reference implementation extraction**
- Scope: Keep `heartbeat()` API signature + theorem docstring + pedagogical simple-reference fallback public in `packages/carl-core/src/carl_core/heartbeat.py`. Move coefficient-pinned reference implementation to `resonance/src/resonance/signals/heartbeat.py`. Public `heartbeat()` delegates to the private impl when available; falls through to the simple-reference when not (so public users still get a working pedagogical reference, sans the benchmarked coefficients).
- Output: ~300 LOC moved to resonance; ~150 LOC public façade + simple-reference fallback.
- Verification: existing heartbeat tests pass both with and without admin-gate. Coefficients referenced in tests become fixture-parameterized so they're not committed to MIT.

**F3. Resonant `joint`-mode extraction**
- Scope: The `joint`-mode cognize path in `packages/carl-core/src/carl_core/resonant.py` (the latent-vector → scalar readout via single EML tree application) is ~80 LOC of commercial IP. Extract to `resonance/src/resonance/geometry/joint_cognize.py` (geometry subpackage is the natural home for latent-space math). Keep `per_dim` cognize + the `Resonant` container + canonical encoding public.
- Output: ~80 LOC moved to resonance; `Resonant._cognize_joint` becomes a private runtime lookup via `admin.load_private("geometry.joint_cognize")`.
- Verification: `per_dim` tests untouched; `joint`-mode tests require admin-gate to run. Document in public Resonant docstring that `joint` mode requires the private runtime.

**F4. EML reward trained-weights extraction**
- Scope: `src/carl_studio/training/rewards/eml.py` contains the initialization heuristics + 7-param trained coefficients that benchmark at +0.972 correlation with PhaseAdaptive. Move numeric constants + init heuristic to `resonance/src/resonance/rewards/eml_weights.py` (the `resonance/rewards/` subpackage already exists with siblings `composite.py`, `coupling.py`, `gate.py`, `load.py`). Keep the structure (depth-3 tree, composition rules, scoring interface) public. Public reward loads with default-uninitialized weights; private runtime populates benchmarked numbers on admin-gate.
- Output: ~150 LOC moved to resonance; public reward falls back to random-init.
- Verification: benchmark script documents both paths. Public CI test validates structure only. Private reproduction test (admin-gated) asserts the +0.972 benchmark number.

**F5. `admin.py` load_private() enhancement** (opportunistic cleanup per §13.2)
- Scope: Try `import resonance.<module_name>` first; fall back to HF dataset download. Preserves distributed-access path while making local dev faster + offline-capable for terminals-team machines.
- Output: ~20 LOC change to `src/carl_studio/admin.py::load_private`.
- Verification: 2 new tests — `test_admin_gate_prefers_local_resonance`, `test_admin_gate_falls_back_to_hf_dataset`.

**F6. Admin-gate pattern formalization** (overlaps with §4.A/A8)
- Scope: Document the pattern. `resonance/__init__.py` pattern (the `resonance` package already exists and acts as `terminals_runtime` in all but name), `admin.load_private()` contract, registration-at-admin-gate-time only.
- Output: `docs/v17_admin_gate_pattern.md`.
- Verification: CI grep check — `grep -rE "^(from resonance|import resonance)" src/carl_studio/ packages/carl-core/` returns only lines inside admin-gated functions.

**F7. CI-level moat enforcement**
- Scope: Add pre-commit hook + GitHub Action that fails the build if any code in `packages/carl-core/` or `src/carl_studio/` imports `resonance` or `terminals_runtime` at module level (only inside `if admin.is_admin():` or equivalent lazy blocks is allowed).
- Output: `.github/workflows/moat-boundary-check.yml` + pre-commit hook.
- Verification: regression test — introducing a top-level `import resonance` fails CI.

**F6. CI-level moat enforcement**
- Scope: Add a pre-commit hook + GitHub Action that fails the build if any code in `packages/carl-core/` or `src/carl_studio/` imports `terminals_runtime` at module level (only inside `if admin.is_admin():` or equivalent lazy blocks is allowed).
- Output: `.github/workflows/moat-boundary-check.yml` + pre-commit hook.
- Verification: regression test — introducing a top-level `import terminals_runtime` fails CI.

**F8. Team F readout**
- Structured readout per §12 format.
- Must include: total LOC extracted, list of private-module landing paths, admin-gate round-trip test evidence, CI-hook proof.

---

### 4.E Policy / IP (Team E)

**E1. Tier split update**
- Output: new `FEATURE_TIERS` key `secrets.managed_vault = Tier.PAID`.
- Rationale: carl.camp hosting the vault service (compliance, audit, rotation) is autonomy-as-a-service.
- Verification: `test_tier_features.py` updated.

**E2. Private-repo boundary audit** (Proposal 8)
- Output: `docs/v17_moat_boundary.md` — documents admin-gate seam, what lives public vs private, policy for new features.
- Deliverable: grep-based CI check ensuring `carl_studio/**` never imports from `terminals_runtime.*` at module-level (only inside admin-gated functions).

**E3. Documentation updates**
- `docs/v17_vault_resolver_chain.md` — user-facing guide for the resolver chain
- `docs/v17_handle_runtime_doctrine.md` — updated post-Vault-unification (supersedes `v16_handle_runtime.md`)
- `docs/v17_training_spec.md` — user-facing TrainingSpec guide
- Update CLAUDE.md with new architecture snapshot
- Update CHANGELOG.md with v0.17.0 section

**E4. Policy process updates** (captured in CLAUDE.md)
- Ban `X or X()` pattern for vaults/stores (add a lint rule)
- Every workstream ships with a README.md in its package root
- `Session` is the canonical user-facing entry point; examples lead with it
- Private-runtime extensions register at admin-gate time only, never at import

**E5. Team E readout**

---

## 5. Sequencing + parallelism

```
DAY 1  ── A1 + A2 (generic Vault + resolver chain)         [Team A]
         + F1 start (constitutional extraction)             [Team F]
         + E1 + E2 (tier + moat audit)                      [Team E]
DAY 2  ── A3 + A4 + A5 (three vault specializations)       [Team A]
         + F2 (heartbeat extraction)                        [Team F]
DAY 3  ── A6 (public resolvers) + A7 (Session)             [Team A]
         + F3 + F4 (resonant joint + EML weights)           [Team F]
         + E3 start (doc stubs)                             [Team E]
DAY 4  ── A8 + A9 (admin-gate + readout)                   [Team A]  ← FOUNDATION MERGE
         + F5 + F6 (pattern doc + CI enforcement)           [Team F]
         + F7 readout                                       [Team F]  ← MOAT MERGE
────────────────────────────────────────────────────────
DAY 5  ── B1 + B2 + C1 + D1                                 [B, C, D parallel]
DAY 6  ── B3 + B4 + C2 + C3 + D2
DAY 7  ── B5 + B6 + B7 + C4 + C5 + D3
DAY 8  ── B8 + C6 + C7 + D4 + E3 finish + E4
DAY 9  ── C8 (Session integration in agent, adopts HandleRuntimeBundle)
         + integration tests + cross-team reconciliation
DAY 10 ── Final regression + readouts + ship
```

**Parallel opportunity.** Team E runs concurrently with A and F from Day 1. Team F runs fully parallel to Team A; their file sets are disjoint (A touches vault/session, F touches constitutional/heartbeat/resonant/eml-rewards). Both converge at Day 4 merge.

**Critical-path gate.** Day 4 merge must clear Foundation + Moat together before Teams B/C/D start. This protects Teams B-D from needing to rebase on either foundation changes or moat moves.

---

## 6. Acceptance criteria (global — all tracks)

**Must pass before ship:**

- [ ] All 3,599 existing tests green (tolerance: ±10 for renames / stripped assertions)
- [ ] 0 pyright strict errors on all new + changed files
- [ ] 0 ruff warnings on all new + changed files
- [ ] Full-suite timing ≤ 80 s (current baseline 72 s; +10 % ceiling)
- [ ] `import carl_studio` imports in ≤500 ms (current baseline measured in Team A day-1)
- [ ] `carl.Session()` is the documented primary entry point for new user-facing examples
- [ ] EML wire format test vectors pass (shared with npm codec)
- [ ] `carl.yaml` fixtures unchanged
- [ ] κ = 64/3 (unchanged)
- [ ] Private-repo audit (Team E) confirms zero IP leakage into MIT surface
- [ ] Every team lead has signed off with a structured readout

**Stretch:**
- [ ] Import time ≤ 300 ms
- [ ] Unified scrubber registry (D4) ships

---

## 7. Risk register

| Risk | P × I | Mitigation | Trigger for rollback |
|---|---|---|---|
| `Vault[H, V]` generic typing — Pyright strict + Pydantic v2 Generic issues | Med × High | Prototype in A1; fallback to ABC + Protocol if Generic doesn't type-check cleanly | Day 1 — Pyright fails → fallback |
| `chat_agent` split breaks private-method test hooks | Med × Med | Preserve all hook points as explicit submodule exports; run test suite after each extraction | Day 6 — ≥5 test failures → revisit split granularity |
| TrainingSpec schema drift from `carl.yaml` | Low × High | Parse `carl.yaml` first, then build Spec. Schema additive only | Day 5 — any yaml fixture fails → revise |
| Resolver chain adds latency (CLI shell-outs) | Med × Low | TTL cache in `resolve()`; circuit breaker per resolver | Day 3 — e2e browser test +>500ms → investigate |
| Session snapshot/restore size explosion | Low × Med | Snapshot only refs + metadata, not values. Property test on serialized size | Day 3 — snapshot >100 KB for empty session → fix |
| Team parallelism creates merge conflicts | Med × Med | Foundation blocks all others; Teams B/C/D/E own disjoint file sets | Day 6 — any conflict requires 2-lead sign-off to resolve |
| `@audited` decorator + existing `probe_call` race | Low × Low | Decorator checks for existing coherence fields; only records if absent | Day 7 — any duplicate step → fix dedup |

---

## 8. Paid vs freemium split — delta for v0.17

### Remains FREE (capability, not autonomy)

- `Vault[H, V]` base + specializations
- All three public resolvers (`env`, `keyring`, `fernet-file`)
- `Session` + `TwinCheckpoint` primitives
- `TrainingSpec` + all adapter translators
- `@audited` decorator + content-addressed steps
- `chat_agent` package surface (unchanged)
- Constitutional ledger + EML primitives (unchanged)

### NEW PAID — `secrets.managed_vault`

carl.camp hosts the vault service: compliance-audited storage, rotation, cross-device sync, team access control. User's vault lives in carl.camp's encrypted storage; local resolver chain hits carl.camp's API. This is autonomy-as-a-service — users don't manage secrets hygiene themselves.

### NEW PAID precedent set — managed resolver registry

Same tier split applied recursively: public resolvers free; carl.camp-managed resolver fleet (with SLA, rate-limit pooling, audit) is paid. Future resolvers (OnePassword-as-a-service, Vault-as-a-service) default to paid when they involve carl.camp compute or network.

### FREE tier guarantees preserved

No FREE key becomes PAID. All capability primitives stay free per the tier doctrine.

---

## 9. Private-repo moat seam

### 9.1 Public (MIT) — carl-studio

- `Vault[H, V]` + resolver chain protocol
- Public resolvers (env, keyring, fernet-file)
- `register_runtime_resolver(admin_token, kind, resolver)` API
- `AdminToken` class + gate pattern in `carl_studio.admin`

### 9.2 Private (BUSL / proprietary) — `/resonance/*` + `terminals-runtime/*`

- Hardware-attested resolvers (HSM, Secure Enclave, TPM)
- Proprietary EML-head fit path (already private per `ttt/eml_head.py` seam)
- Proprietary signing-tier 2 / hardware-tier signing
- Whatever the user chooses to move to `/resonance/*` based on Team E audit

### 9.3 Policy enforcement

- CI check: `grep -r "from terminals_runtime" src/carl_studio/ packages/carl-core/` → must return zero top-level matches (only inside admin-gated functions)
- Pre-commit hook enforces the same
- Every new feature reviewed against the rule before PR merge

### 9.4 Digital-twin + Terminals OS convergence implications

1. **`carl_core.vault` is the seed for `terminals_core.vault`.** Design it such that a future `terminals-core` can absorb the module wholesale, renaming the `carl_` prefix. No carl-specific constants in the primitive.
2. **`Session` maps to a Terminals OS "tab" / "task."** A Terminals OS task *is* a Session with scoped chain + tools. Our Session design should not preclude multi-session supervision.
3. **`TwinCheckpoint` is the portability primitive.** A twin's state = its chain + vault refs + memory layers. Content-addressed ensures portability.
4. **Admin-gate seam is load-bearing for moat migration.** When terminals-runtime lands publicly (if it ever does, in distant future), the seam pattern generalizes to a plugin registry.

---

## 10. Policy + process updates to carry forward

All updates to be merged into `CLAUDE.md`:

1. **Ban `X or X()` pattern for anything with `__len__`.** Use `X if X is not None else X()`. (Cause of the v0.16.1 silent-divergence bug.) Add a CI grep check.
2. **Every vault subclass must register at least one resolver on construction** (even if it's a no-op). Enforces the "resolver chain is first-class" invariant.
3. **`Session` is the canonical user-facing entry point.** All new docs, examples, and README snippets lead with Session.
4. **Private-runtime extensions register at admin-gate call time only.** Never at module import. No `import terminals_runtime` anywhere in `carl_studio/**`.
5. **Every workstream package ships with a README.md.** Documents the package's shape, contract, and test coverage expectations.
6. **God-class register in CLAUDE.md.** Any file > 1,500 LOC gets flagged; >2,000 LOC gets a deadline for split.
7. **Property-based tests for invariants.** Every new primitive with a lifecycle (put/resolve/revoke/expire) gets a hypothesis test.
8. **Content-addressed audit trail.** Once D3 lands, every chain step is addressable. This is load-bearing for v0.18+ (replay, cross-process chain merging).

### 10.9 CLI–AST isomorphism (new doctrine — Tej, 2026-04-21)

Command namespace mirrors module namespace. `carl <subpackage> <module> <verb>` ↔ `carl_studio.<subpackage>.<module>.<ClassName>.<verb>`.

**Why.** Tej's intuition: "I would imagine theres some value there in codebase composability and interpretability if the AST are both clean and both representable as a graph/tree in the first place." This is correct and has prior art in kubectl / helm / hugo — nested verb surfaces that mirror the codebase tree.

**What it buys us:**

- **Tab-completion = module exploration.** Users discover capabilities by walking the tree.
- **Help text auto-derives from docstrings.** One source of truth.
- **Refactor safety.** Rename module → rename command (one mechanical pass); lint rule enforces the match.
- **Testing simplicity.** Test the method; CLI is a thin dispatch shim.
- **Terminals OS alignment.** When carl's CLI eventually merges into a Terminals OS command space, the namespace-graph carries over cleanly.

**v0.17 policy:**

- Any NEW CLI command added by Teams A-F (there shouldn't be many) follows the principle.
- Teams C (chat_agent split) and F (moat extraction) that touch CLI boundaries make their new commands conform.
- **No bulk reorg of existing 94 commands this release** — that's a v0.18 dedicated UX sprint.

**v0.18 policy (preview):**

- Bulk CLI reorg: `carl train → carl training run`, `carl observe → carl observe live`, etc.
- Automated lint rule: `cli/<path>.py` must declare commands under a typer app whose name matches the module tree.
- Dedicated graph-export tool: `carl meta commands --as-tree` prints the command AST next to the module AST (diff must be empty).

### 10.10 Opportunistic-cleanup policy (new — Tej, 2026-04-21)

When a team touches a file in flight, they MAY do minor obvious cleanups in the same PR: unused imports, dead `_ = X` keep-lines, obvious typos, missing `from __future__ import annotations`. Scope boundary: cleanups must be <10 LOC per file and must not change public API. Anything bigger goes in its own ticket.

---

## 11. Digital Twin + Terminals OS alignment matrix

| Proposal | Digital twin win | Terminals OS convergence win |
|---|---|---|
| `Vault[H, V]` | Twin owns refs, not values — portable across devices | Becomes `terminals_core.vault` wholesale |
| Resolver chain | Twin can resolve against user's preferred backend | Generalizes to OS-level secrets daemon integration |
| `Session` | Primitive for twin state boundary | Maps 1:1 to Terminals OS task/tab |
| `TwinCheckpoint` | Explicit checkpoint/restore for digital twin | Portable work surface across Terminals OS instances |
| `TrainingSpec` | Twin's training workloads are portable | Unified workload schema for terminals-compute (future) |
| `chat_agent/` split | Agent's core becomes a terminals-OS citizen | Cleaner migration to terminals-agent runtime |
| `@audited` + content-hash | Twin's behavior is cryptographically verifiable | Terminals OS kernel can verify chain integrity |
| Admin-gate seam | Private proprietary twin features plug in cleanly | Terminals-runtime → future first-class plugin system |

---

## 12. Readout format (team lead sign-off)

At end of each workstream, the team lead produces a one-page readout in `docs/v17_team_<X>_readout.md`:

```
# v0.17 Team <X> readout

## Scope honored
- [list of atomic tasks completed]

## Scope cut (with rationale)
- [anything deferred]

## Metrics
- Files changed: <N>
- LOC delta: +<added> / −<removed> / net
- Tests added: <N>
- Tests regressed: <N> (must be 0)
- Pyright errors: 0 / <N>
- Ruff warnings: 0 / <N>
- Full-suite timing delta: +/−X%

## Blast radius (actual vs predicted)
- Predicted: <from §1.4>
- Actual: <measured>

## Decisions made
- [key architectural calls + rationale]

## Handoff to next workstream
- [what the next team needs to consume]
```

Orchestrator (Claude/me) produces the integrated executive readout at ship time covering:
- Scope honored / cut (per team)
- Full-suite outcome
- Risk register outcome (what fired, what didn't)
- UX friction delta (dev wiring lines: 4 → 1)
- Tier surface changes
- Private-repo audit confirmation
- Recommendations for v0.18

---

## 13. Decision log (LOCKED 2026-04-21 with Tej)

1. **Moat extraction scope (Team F).** ✅ **ALL FOUR IN.** Tej: "let's do that now if you think worth gating and keeping in the moat category especially the coefficients and invariants." Confirmed: `constitutional`, `heartbeat`, `resonant.joint`, `eml-rewards-weights` — all ship v0.17.
2. **Heartbeat public fallback semantics.** ✅ Keep a pedagogical simple-reference fallback public; private runtime contributes pinned coefficients + reference implementation. (Decided as Claude's rec per Tej default.)
3. **TwinCheckpoint format.** ✅ Content-addressed JSON. Battle-tested pattern (git objects are content-addressed). Binary format deferred until/unless snapshot sizes demand it.
4. **Which adapter migrates first (Team B)?** ✅ TRL.
5. **Scrubber registry (D4).** ✅ **DEFER.** Explicitly out of v0.17.
6. **`Session.__init__` signature.** ✅ Claude decides: `Session(user: str | None = None, *, chain: InteractionChain | None = None, secret_vault: SecretVault | None = None, resource_vault: ResourceVault | None = None, headless_browser: bool = True)`. All kwargs after `user` are BYOC-optional; defaults construct fresh. `user` positional for `carl.Session("tej")` ergonomics.
7. **Terminals OS naming.** ✅ Keep `carl_core` for v0.17. Rename is a separate mechanical pass later.
8. **CLI fatigue.** ✅ Bulk UX pass deferred to v0.18. **NEW principle added (Tej's insight):** opportunistic cleanups as teams touch CLI files, AND codify **CLI-AST isomorphism** — command namespace should mirror module namespace. See §10.9.
9. **Private repo readiness.** ✅ **Resolved by audit.** Local private repo is `/Users/terminals/Documents/agents/models/resonance/` (package name: `resonance`, v0.1.0, depends on `carl-studio>=0.2.0`). Target subpackages already exist: `resonance/eml/`, `resonance/ttt/`, `resonance/rewards/`, `resonance/geometry/`, `resonance/signals/`, `resonance/deployment/`. Team F ships without repo setup blockers.

### 13.1 Extraction landing map (concrete file paths)

| Moat file (public, before) | Target path (private, after) | Notes |
|---|---|---|
| `packages/carl-core/src/carl_core/constitutional.py` (556 LOC, FULL move) | `resonance/src/resonance/signals/constitutional.py` | Lands in `resonance/signals/` since constitutional ledger IS a signal-domain primitive. Thin `ConstitutionalLedgerClient` remains in carl-core exposing the public `genesis/append/verify_chain` surface. |
| `packages/carl-core/src/carl_core/heartbeat.py` (~300 LOC of pinned impl) | `resonance/src/resonance/signals/heartbeat.py` | Public heartbeat.py keeps API + theorem docstring + pedagogical simple-reference fallback. Private side has coefficient-pinned reference impl. |
| `packages/carl-core/src/carl_core/resonant.py` — joint-mode cognize path (~80 LOC) | `resonance/src/resonance/geometry/joint_cognize.py` | Geometry subpackage is the natural home for latent-space math. `per_dim` cognize stays public; `joint` dispatches to private via admin-gate. Confirms Tej's note: "resonant file is serving Resonant, not the actual resonance logic." |
| `src/carl_studio/training/rewards/eml.py` — trained weights + init heuristics (~150 LOC) | `resonance/src/resonance/rewards/eml_weights.py` | `resonance/rewards/` already exists with sibling `composite.py` / `coupling.py` / `gate.py` / `load.py`. Perfect landing. Public reward keeps structure + random-init default; private side contributes benchmarked numbers. |

### 13.2 `admin.py` enhancement (opportunistic cleanup — Team F side-effort)

Current `load_private()` downloads from `wheattoast11/carl-private` HF dataset. For local dev with `resonance` pip-installed, we should try direct import first:

```python
def load_private(module_name: str) -> Any:
    if not is_admin():
        raise ImportError(...)
    # (1) Try local resonance package (local dev, terminals-team machines)
    try:
        import importlib
        return importlib.import_module(f"resonance.{module_name}")
    except ImportError:
        pass
    # (2) Fallback: HF dataset download (distributed access)
    from huggingface_hub import hf_hub_download
    ...
```

Keeps the distributed-access path; adds fast-path for local dev.

---

## 14. Appendix — UX/IP audit findings

Measured 2026-04-21 via background subagent. Key findings:

### 14.1 USER surface (operator cognitive load)
- **94 top-level `@*_app.command` registrations** across 20 `cli/*.py` files
- **~110+ effective command invocations** including `add_typer` mounts
- **22 flow ops** in `cli/operations.py` OPERATIONS registry + 6 injected prompt ops
- **20 MCP tools** exposed via `mcp/server.py`
- **Heavy-fat command.** `carl train` alone declares **19 `typer.Option`s**. Minimum viable invocation still requires 3-5 flags. **Flag for v0.18+ UX pass.** Not fixed in v0.17 (scope discipline).

### 14.2 DEV surface
- **Wiring lines today:** 4 lines + 3 imports to get `CARLAgent` + full v0.16.1 toolkits (via `HandleRuntimeBundle.build` pattern).
- **Wiring lines target (Session):** 2 lines + 1 import (`with carl.Session() as s: s.register_tools_to(agent)`).
- **Public `__all__` surface:** 7 eager + 13 lazy (`carl_studio/__init__.py:23-35,60-78`). Session fits cleanly as a new top-level export.
- **Top-level `carl_studio/` modules:** 42 `.py` files (6 user-facing, 36 internal plumbing) + 24 subpackages.

### 14.3 IP moat boundary
- **Architecture is clean** — admin-gate + lazy-import + graceful-degrade verified at 10 call sites (`ttt/eml_head.py:54-66`, `admin.py:32-62`, etc.).
- **What's on the public side of the line is the problem.** See §0 moat leak table.
- **HandleRuntimeBundle live gap:** `grep HandleRuntimeBundle src/carl_studio/chat_agent.py` returns 0. The canonical wiring bundle I shipped in v0.16.1 is consumed only by `tests/test_handle_bundle.py`. **Team C MUST adopt during the split.**
- **`carl_studio/resonant_store.py`** correctly MIT (thin envelope + file I/O); moat is in the `Resonant` class joint-mode path below it.
- **`src/carl_studio/fsm_ledger.py`** correctly MIT (thin glue over `carl_core.constitutional`); moat moves with the core file.

### 14.4 Admin-gate pattern verification
Verified working at:
- `src/carl_studio/ttt/eml_head.py:54-66` (private fit path)
- `src/carl_studio/ttt/slot.py:49,92` (SLOT optimizer)
- `src/carl_studio/training/lr_resonance.py:82` (resonance scheduler)
- `packages/carl-core/src/carl_core/coherence_observer.py:61` (Claude observer)
- `packages/carl-core/src/carl_core/frame_buffer.py:134` (frame buffer)

Pattern correctly applied but not *uniformly* — Team F's work is to extend it to constitutional / heartbeat / resonant.joint / eml-rewards.

---

## 15. Source data

- **Blast radius metrics** — measured 2026-04-21 via `grep -r` counts across `src/`, `packages/`, `tests/`.
- **God-class LOC** — measured via `find ... | xargs wc -l`.
- **Test baseline** — 3,599 passing, 1 skipped, 0 failed, 72 s full suite (minus UAT + heartbeat fixture-collision).
- **Proposal origins** — previous session's "unrestricted architectural change" conversation, refined against measured blast radius.
- **Philosophy reference** — `packages/carl-core/src/carl_core/tier.py:16-24` tier doctrine ("gate on autonomy, not capability").
