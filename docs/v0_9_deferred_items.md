---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9.0
---

# v0.9.0 Deferred Items

Record of features discussed, specced, or conjectured during the v0.9.0
EML×CARL integration swarm but NOT shipped in this release. Organized by
category and priority so future sessions can pick up without context loss.

This is the durable cross-session record; temporary session todos live
in TaskCreate only. When an item here is completed, either delete its
entry or move it under `## Category 0 — Shipped since v0.9.0`.

## Category 0 — Shipped since v0.9.0

- **2026-04-21 — `cli/resonant.py` with `publish` / `list` / `whoami` /
  `eval`** (commit pending). Moved out of §1.1. Remaining in §1.1:
  `carl resonant fit` (needs admin-gated private backend), `carl
  resonant compose` (utility, low priority), `carl marketplace buy`
  (requires carl.camp purchase endpoint to land first), `carl ledger
  today` (cli/contract.py has `constitution status` — ledger read
  command deferred).
- **2026-04-21 — `@terminals-tech/emlt-codec@0.1.0` published to
  npm.** 82/82 tests, ESM + CJS dual, byte-parity with Python reference.
- **2026-04-21 — `carl_core.signing` public software-tier helpers**
  (`sign_tree_software`, `verify_software_signature`, platform
  countersig pair). 11 tests.
- **2026-04-21 — `docs/eml_signing_protocol.md` §5.1 + §5.2**
  locking `X-Carl-User-Secret` transmission and
  `sig_public_component = sha256(user_secret)[:16]` fingerprint.
- **2026-04-21 — `carlcamp_eml_integration_brief.md` refresh**
  replacing stale ed25519 primary-path refs with HMAC software tier.

## Category 1 — Near-term patches (v0.9.x / v0.10.0)

### 1.1 CLI subcommands specced in T10 but not wired
Seven CLI commands were fully specified in `docs/summercamp_eml_ux.md`
and `docs/hardware_interface_spec.md` but only `carl contract
constitution [genesis|verify|evaluate|status]` actually ships in v0.9.0.

- ~~`carl resonant eval`~~, ~~`carl resonant list`~~,
  ~~`carl resonant publish`~~, ~~`carl resonant whoami`~~ —
  SHIPPED 2026-04-21, see Category 0
- `carl resonant fit <dataset>` — train an EML tree on provided data
  (needs admin-gated backend; follow `ttt/eml_head.py` pattern)
- `carl resonant compose <tree_a> <tree_b>` — magma composition with
  depth-4 guard (utility wrapper around `carl_core.resonant.compose_resonants`)
- `carl marketplace buy <resonant_id>` — x402 settlement for a
  downloaded resonant (blocked on platform purchase endpoint)
- `carl ledger today` — today's constitutional ledger blocks (read
  surface for the Supabase ledger; complements
  `carl contract constitution status`)

Implementation pattern: add `cli/resonant.py`, `cli/marketplace.py`, `cli/ledger.py`
mirroring `cli/contract.py`. Register in `cli/__init__.py`. Reuse existing
`admin.is_admin()` + `consent_gate` machinery.

**`carl resonant publish` wire contract (v0.9.x, locked 2026-04-21).**
Confirmed with the carl.camp platform agent; the upload MUST match
`docs/eml_signing_protocol.md` §5.1 exactly.

```
POST {carlcamp_base}/api/resonants
Content-Type: application/octet-stream
Authorization: Bearer {supabase_jwt}
X-Carl-User-Secret: {base64(contents_of_~/.carl/credentials/user_secret)}
X-Carl-Input-Dim:   {tree.input_dim}
X-Carl-Output-Dim:  {readout.shape[0]}
X-Carl-Projection:  {base64(projection.tobytes())}
X-Carl-Readout:     {base64(readout.tobytes())}

<body: raw .emlt envelope = codec_impl.encode(tree, sig=sign_tree_software(tree.to_bytes(), secret))>
```

CLI obligations for the implementer of `cli/resonant.py::publish`:

- Read `~/.carl/credentials/user_secret` (mode 0600); generate with
  `os.urandom(32)` on first run if absent; never log the raw bytes.
- Compute `sig_public_component = sha256(user_secret)[:16].hex()`
  and show the first 8 chars on `carl whoami` and the
  publish-confirmation prompt so the user recognizes their identity.
  **Do NOT** send the fingerprint as a header — server re-derives
  from the body.
- Standard base64 for the header (NOT url-safe); matches the
  `@terminals-tech/emlt-codec` server-side decode.
- Add `X-Carl-User-Secret` to the CLI HTTP client's redaction list so
  verbose logs and tracebacks never surface it.
- On 402 response, surface the platform's x402 payment body; the
  publish path is free-tier, 402 indicates tier misconfiguration.

### 1.2 EML-native Adam optimizer
Team ι proposed replacing Adam's `m̂/(√v̂ + ε)` update with
`exp(log m̂ - 0.5·log v̂)` for full EML coordinates. This would be the
first optimizer that is itself an EML tree. Lives in
`carl_core.optim.eml_adam` (new module). Estimated LOC: ~120.

### 1.3 Jointly-trained EML coherence gate
Team T3 shipped the smooth gate `eml(R, min_R) > τ` with a constant `τ`.
Team ι proposed making the gate itself a learnable EML tree:
`gate(Φ, history) = eml(f(Φ), g(history))` jointly trained with the
policy. `gating.py` gains a `LearnedEMLGate` class that holds an
`EMLTree` and steps on reward advantage.

### 1.4 Constitutional tree mutation guards
Team λ specified but did not build:
- **Behavioral drift snapback** — reject any ΔB update that reduces
  chain `phi` below DEFECT_THRESHOLD. Add `C_drift` subtree to genesis
  constitution that evaluates drift velocity.
- **Emergency halt subtree** — a special C node that, when evaluated to
  above a threshold, immediately freezes the FSM. Analog of Ethereum's
  social-fork escape hatch.
- **C_mut subtree** — per-policy mutation-admission circuit that bounds
  Adam step size to `|Δw| ≤ max_step`. Currently only a comment in
  `fsm_ledger.py`; needs actual EML-tree encoding of the bound.

### 1.5 Integration smoke test for the live heartbeat loop
Heartbeat + OptimizerStateStore + ConstitutionalLedger + EMLReward all
exist as separate primitives; no test exercises them together end-to-end
in a GRPO training step. Add `tests/test_heartbeat_integration_smoke.py`
that runs 100 training ticks, persists state, verifies trajectory
exhibits bounded oscillation, and checks the ledger grew correctly.

### 1.6 Cross-WASM engine bit-identity empirical validation
T8 proved determinism *by construction* (softfloat uses only IEEE 754
basic ops). Actual A/B test of `exp_det(x).to_bits()` across wasmtime,
wasmer, V8, Firefox, Safari is deferred. Harness in
`crates/terminals-wasm/tests/cross_engine_bit_identity.rs`.

### 1.7 Spectral mode detection hooked into training
`heartbeat.detect_resonant_modes()` exists but is not wired into the
training loop to auto-detect when the system exits the edge-of-chaos
band. Add a `ResonantModeProbe` callback that runs every N ticks.

### 1.8 simulate_R_squared consumer integration
T4 shipped `simulate_R_squared` in `resonance/signals/observer_sim.py`
but no carl-studio code calls it. Eventually replace internal
`trace.kuramoto_R()` call sites that square the result (search for
`R ** 2`, `R*R` patterns) with direct `simulate_R_squared` calls —
preserves depth-3 EML structure end-to-end without the spurious √.

### 1.9 Documentation trim pass
- `carl-studio/CLAUDE.md` currently 423 lines (over T11's 300-line budget).
  T11 respected additive-only rule and did not trim. Candidate for
  migration to `docs/architecture_history.md`: trailing "Historical
  design docs", "Anti-pattern catalog", and "Mental model" sections.
- `zero-rl-pipeline/CLAUDE.md` at 416 lines; similar trim candidate.

### 1.10 Test suite pyright hygiene
Pyright strict mode produces many unknown-type warnings inside pytest
tests (`pytest.approx` returns `ApproxBase` without type args). Not a
correctness issue but pollutes IDE diagnostics. Add a
`tests/_typing.py` with typed wrappers if team capacity allows.

## Category 2 — Platform work (carl.camp agent scope)

These land on the **carl.camp platform agent** (different session).
Spec already shipped at `docs/carlcamp_eml_integration_brief.md`
(1896 words). Brief handoff prompt for that agent lives at
`docs/carlcamp_agent_handoff_prompt.md`.

- Supabase schema: `resonants`, `constitutional_ledger_blocks`,
  `eml_marketplace`, `training_runs.eml_head_id` column
- RLS policies + indices + check constraints per brief §2.a
- `POST /api/resonants/purchase` x402 settlement endpoint (§2.b)
- AgentCard extensions: `supports_eml`, `resonant_ids`,
  `eml_policy_hash`, `eml_depth_cap` fields (§2.c)
- Discovery filter endpoint: "agents offering resonants for domain X"
- MCP tools: `eml_evaluate`, `eml_fit_request` with JSONSchema (§2.d)
- Provider webhook `POST /api/hooks/run_complete` (§2.e)
- Signature verification gate before any `.emlt` insert
- Rate limits: 10/hr free, 100/hr paid (depth ≤ 4 enforced server-side)
- Grafana board: resonant fitness trajectory + ledger volume
- Prometheus counters: fits/hr, evals/hr, purchases/hr

## Category 3 — Research + papers

- **Halo2/PLONK SNARK circuit for EML evaluation.** Team λ estimated
  ~800 constraints per depth-4 proof, ~40ms prove / ~8ms verify, ~150KB
  proof — fits one EIP-4844 blob. Circuit not implemented. Target
  venue: `zkml` workshop or arXiv.
- **EML-as-HVM2-interaction-net compilation path.** Team β flagged the
  `EmlNode → γ(exp, log)` Lafont mapping as unbuilt territory (~150 LOC
  estimate). Would give EML trees shared-memory parallel evaluation on
  HVM2. Would justify a short companion paper.
- **Lean 4 formalization of the Standing Wave Theorem** (ι Dim 3a).
  Paper's Appendix B states the theorem with proof sketch; Lean
  formalization strengthens from "proof sketch" to "proof verified".
  Odrzywolek's original Lean attempt failed because `Complex.log 0 = 0`
  (junk total function) — Lean 4 Mathlib may have evolved enough.
- **Bitc / DMC / IRE paper updates** to cite EML as the third
  realizability witness. Cross-references exist in the new
  `eml-symbolic-witness.md` paper but the originals have not been
  updated to close the triangle. Small edit to each of the three.

## Category 4 — Hardware (long-horizon)

All items specced in `docs/hardware_interface_spec.md` but hardware is
not yet fabricated. The software surfaces land in v0.10.0 or v0.11.0.

- USB device: bootable image + hardware wallet + Titan-style 2FA + LEDs
- Wristband: BLE GATT PTT + mic + haptic + biometric + cert profile
- Baseball cap: Arduino bone-conduction + WebMIDI sensor input
- Inland 37 metric monitor integration (Bluetooth LE)
- `carl admin attest-device <serial>` CLI
- `/dev/terminals-usb` Linux udev rule
- WebSerial browser shim with Firefox/Safari base32-code fallback
- boot-from-USB lite mode (Tauri signed, v0.10.0 target)
- boot-from-USB full Alpine image (v0.11.0 target)
- Ed25519-signed A2A with monotonic replay counter
- Resonant ↔ device binding layer (Resonant subscribes to device
  stream, emits to device actuator; <10ms round-trip target)

## Category 5 — Economic + governance (speculative / conjectural)

These survive from the μ (matter-antimatter governance) review as
either honest metaphor-with-structure or research directions —
explicitly NOT shipped as governance primitives. The paper's §1.5
non-claims disclaims them. Listed here for future exploration, not
for execution.

- Log-barrier reward shape as a named primitive (Nesterov-Nemirovski
  interior-point lineage). Ship as `training/rewards/log_barrier.py` if
  any project finds it useful; do not market as "governance".
- Tropical algebra (max, +) semiring primitives for option pricing /
  routing — a legitimate mathematical family with real usage; deferred
  until a concrete use case emerges.
- On-chain inscription of the constitutional ledger head hash to
  Ethereum L2 or Solana. Architecture-only in λ; cryptography crate
  already in optional extras.
- NFT representation of Resonant identities. Would use the existing
  `Resonant.identity: sha256` hash as the token metadata.
- Agent reputation scoring via ledger replay weights. Conjecture;
  needs a use case.
- Derivatives markets for agent capabilities (Tej's tokenomics
  thesis). Conjecture; out of scope for v0.x.

## Category 6 — UX surface (spec → ship gap)

Listed in T10's `summercamp_eml_ux.md` but not built. Each lives on
carl.camp platform agent's backlog and is cross-referenced in
§Category 2 above.

- Heartbeat dashboard (three-pane: pulse / "what CARL noticed today" /
  suggestions)
- Drag-drop magma-tile composition UI
- "My resonance fell out of sync — help me recalibrate" dialog flow
- Marketplace browse/purchase/deploy UI
- First-run fit-a-sine 30-second onboarding
- Signing + publishing UX for a freshly-trained Resonant
- CLI parity: every UX action has a `carl` command (Category 1.1
  unblocks this)

## Notes on handling this file

- When you ship an item, move its entry to Category 0 with a date + PR
  reference. Do not silently delete — the record is the cross-session
  memory.
- When a deferred item becomes obsolete (superseded by a different
  design), move it to `archive/deferred_items_obsolete.md` with a
  one-line reason. Do not silently delete.
- When a new item gets deferred, append to the most specific category.
  Do not create new categories without discussing.
- Keep entries tight: 1-3 sentences each. This is an index, not a
  design doc. Link out to design docs where they exist.
