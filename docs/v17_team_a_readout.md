---
last_updated: 2026-04-21
author: Claude Opus 4.7 (1M context) + Tej Desai
applies_to: v0.17.0-planned
classification: internal — Team A Foundation Sprint readout
---

# Team A Readout — Foundation Sprint

## Scope honored

All four Foundation Sprint days delivered:

- **Day 1 (ae9316b):** `Vault[H, V]` generic base + resolver chain + `HandleRef` Protocol. 26 unit tests including 3 hypothesis property-based invariants + thread-safety under concurrent put/revoke.
- **Day 2a (ace7a2e):** `SecretVault` migrated to `Vault[SecretRef, bytes]` — `secrets.py` 481 → 258 LOC (−223). `ResourceVault` migrated to `Vault[ResourceRef, Any]` with closer-on-revoke preserved — `resource_handles.py` 294 → 197 LOC (−97).
- **Day 2b (ebbb611):** `DataVault` migrated to `Vault[DataRef, bytes]` — `data_handles.py` 597 → 315 LOC (−282). Rich offset-addressable `read(offset, length)` + lazy sha256 + stream buffering preserved.
- **Day 3 (6254cbd):** Three public resolvers (`EnvResolver`, `KeyringResolver`, `FernetFileResolver`) in `carl_studio.handles.resolvers` + `Session` + `TwinCheckpoint` content-addressed snapshot/restore.
- **Day 4 (this commit):** Admin-gate seam formalized — `admin.py::load_private` local-resonance-first fast path + `AdminToken` capability proof + `Vault.register_runtime_resolver`. Test coverage for all three.

## Scope cut (with rationale)

- **`_VaultEntry` public export** — intentionally not exported from `carl_core.vault.__all__`; subclasses access it via class-level extra-dict without needing the type visible. Keeps the public surface minimal.
- **Session API on `carl_studio.__init__`** — Session lives at `carl_studio.session.Session` but we did NOT elevate it to the top-level namespace (`carl.Session`). Defer to the bulk `carl_studio/__init__.py` refresh in the next session or v0.18 — mixing namespace churn with architectural work invites merge pain.
- **`HandleRuntimeBundle` deprecation path** — we broke the circular import by dropping its eager re-export from `carl_studio/handles/__init__.py`. The class itself lives on; callers import it explicitly. Full deprecation can land in v0.18 when Teams B/C migrate away from it.

## Metrics

| Metric | Value |
|---|---:|
| Files changed | 10 new + 5 modified |
| LOC delta | +2,410 / −936 / net +1,474 |
| Tests added (net new) | 69 |
| Tests regressed | **0** |
| Full-suite count | **3,668 pass + 2 skip** (up from 3,599 baseline + 0 skip) |
| Full-suite timing | 75 s (within plan's +10% ceiling of 80 s) |
| Pyright strict errors on new/modified files | **0** |
| Ruff warnings on new/modified files | **0** |

## Blast radius (actual vs predicted)

| Symbol | Predicted files | Actual files touched |
|---|---:|---:|
| Vault unification (all handle classes) | ~50 | 8 (three vault specializations + base + 3 tests + 1 test-helper) |
| Session (additive) | 3 new + opt-in | 5 new (session.py + resolvers.py + 2 tests + admin seam) |
| Admin-gate seam | 3 | 4 (admin.py + vault.py + 1 new test + conftest tweak via Team F) |

Actual blast was **narrower** than predicted because existing vault tests plus downstream toolkits (DataToolkit, BrowserToolkit, SubprocessToolkit, HandleRuntimeBundle) used only the public vault API — refactoring internals was transparent to them.

## Decisions made

1. **`_error_prefix` + `_error_class` + `_err()` on base Vault.** Required to preserve subclass-specific error code taxonomies (`carl.secrets.*`, `carl.data.*`, `carl.resource.*`) AND `pytest.raises(SecretsError)` matchers. Caught via a failing test; fixed before commit. Pattern scales to arbitrary future vault specializations.
2. **HandleRef Protocol with @property read-only members.** Writable-attribute Protocol form fires `reportInvalidTypeArguments` on frozen Pydantic models + frozen dataclasses under Pyright strict. @property form lets both satisfy the Protocol. Cost: slightly more verbose Protocol body; benefit: clean strict-mode narrowing for all frozen subtypes.
3. **Session is ADDITIVE, not replacing.** Data showed 237 `InteractionChain()` occurrences across 48 files — a forced migration is prohibitive. Session wraps a chain; tests that want raw primitives keep direct `InteractionChain()`. Documented in the Session docstring.
4. **Resolver cache TTL is opt-in per-subclass.** Default `_resolver_ttl_s = None` (no caching). `SecretVault` / `DataVault` / `ResourceVault` inherit the default. A test vault uses `_resolver_ttl_s = 1` to exercise the caching path. Opt-in because a stale cache on a secret is a security risk — callers enable caching only when they've reasoned about staleness.
5. **Circular-import resolution in handles/__init__.py.** Dropped eager `HandleRuntimeBundle` re-export because it transitively imported the full CU stack, which clashed with Session's own CU imports. Explicit import from `carl_studio.handles.bundle` preserved as the canonical path.
6. **TwinCheckpoint does NOT serialize values.** Only ref descriptors (metadata). Values live behind resolvers; the checkpoint is deliberately metadata-complete, value-empty. `no-value-in-snapshot` test asserts this invariant.
7. **AdminToken verification is duck-typed.** The Vault (carl-core) cannot import AdminToken (carl-studio). `register_runtime_resolver` accepts `admin_token: Any` and calls `.verify()` if present. Documented as a visibility marker; the real moat enforcement lives in the CI hook (F7).
8. **`admin.load_private` supports dotted paths via local resonance only.** HF dataset fallback handles flat module names. Dotted paths (`signals.constitutional`) need the local package. Documented in the docstring.

## Handoff to next workstreams

Everything Teams B/C/D need from Foundation is shipped. Specifically:

- **For Team C (`chat_agent` split + Session adoption):** Import `carl_studio.session.Session`. `Session.register_tools_to(agent.tool_dispatcher)` replaces the current manual toolkit wiring. `Session.snapshot() / restore()` unblocks the digital-twin checkpoint feature.
- **For Team B (`TrainingSpec`):** Uses the audit chain; the `@audited` decorator from Team D will wrap adapter translate calls when it lands.
- **For Team D (`@audited` + content-hash):** Content-hash infrastructure already exists in `TwinCheckpoint.build` — the same canonical-JSON + sha256 pattern extends to per-step hashes.
- **For Team E (policy docs):** The `_error_prefix` / `_error_class` pattern is now canonical for ALL future vault specializations — add to CLAUDE.md policy.
- **For Team F (F2-F7 continuation):** The `AdminToken` + `register_runtime_resolver` seam is ready. F3 (Resonant joint-mode) → `resonance/geometry/joint_cognize.py` should register via this pattern.

## Canonical usage examples (for docs + CARL system prompt)

**Basic handle runtime:**

```python
import carl

with carl.Session("tej") as s:
    s.register_tools_to(agent.tool_dispatcher)
    tools_for_anthropic = s.anthropic_tools()
    # ... pass tools=tools_for_anthropic to the Anthropic API
```

**Resolver chain (public backends):**

```python
from carl_studio.handles.resolvers import (
    EnvResolver, KeyringResolver, FernetFileResolver,
)

vault = s.secret_vault  # (or construct SecretVault() directly)
vault.register_resolver("env", EnvResolver())
vault.register_resolver("keychain", KeyringResolver())
vault.register_resolver("vault", FernetFileResolver())
```

**Digital-twin checkpoint:**

```python
import json

ckpt = s.snapshot()
(Path.home() / ".carl" / "twins" / "tej-2026-04-21.json").write_text(
    json.dumps(ckpt.model_dump(), default=str, indent=2)
)
# ... later:
data = json.loads(Path("tej-2026-04-21.json").read_text())
from carl_studio.session import TwinCheckpoint, Session
restored = Session.restore(TwinCheckpoint.model_validate(data))
```

**Admin-gate resolver registration (private runtime entry point):**

```python
# Inside resonance/signals/hardware.py (admin-gated private runtime)
from carl_studio.admin import issue_token

def install_hardware_resolvers(vault: SecretVault) -> None:
    token = issue_token()  # raises if admin gate locked
    vault.register_runtime_resolver(
        "hardware-attested",
        hw_resolver,
        admin_token=token,
    )
```

## Known follow-ups (not blockers)

- **F2 (heartbeat extraction) running in parallel.** Background agent dispatched; will integrate on completion.
- **DataToolkit + HandleRuntimeBundle adoption in `chat_agent.py`** (Team C scope) — the live-gap flagged in the UX/IP audit. Not Team A's to fix; Team C splits chat_agent.py and adopts the bundle during the split.
- **CI moat enforcement hook (F6/F7)** pending Team F completion. Until it lands, the moat discipline is enforced by convention + the IP-audit subagent pass.
- **Session namespace elevation** — post-v0.17, `carl.Session` should alias `carl_studio.session.Session` via `carl_studio/__init__.py` update.

## Sign-off

Foundation sprint ships. Teams B / C / D can start the next session; file sets are disjoint from Team A's so concurrent work is safe. Team F F2 continues in background.
