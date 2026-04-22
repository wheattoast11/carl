---
last_updated: 2026-04-21
author: Claude Opus 4.7 (1M context) + F1/F2/F3/F4 subagents + Tej Desai
applies_to: v0.17.0-planned
classification: internal — Team F Moat Extraction Sprint readout
---

# Team F Readout — Moat Extraction Sprint

## Scope honored

All six extraction + enforcement tasks delivered:

- **F1 (6f7ca85):** Constitutional ledger → `resonance/signals/constitutional.py` (556 LOC moved). Thin `ConstitutionalLedgerClient` stub in carl-core; public data-surface (`LedgerBlock`, `encode_action_features`, `signing_bytes`/`block_hash`, `LedgerBlock.verify`) preserved for `@terminals-tech/emlt-codec` npm parity. The moat is the genesis/append/sign lifecycle, not the wire format.
- **F2 (b3b1f12):** Heartbeat pinned-coefficient reference → `resonance/signals/heartbeat.py` (~300 LOC moved). Standing Wave Theorem docstring + API + pedagogical simple-reference fallback preserved public. `HeartbeatConfig` hyperparameter defaults intentionally stay public — they're theorem-referenced, not calibration constants.
- **F3 (50da47f):** Resonant `joint`-mode cognize → `resonance/geometry/joint_cognize.py`. `per_dim` mode + `Resonant` container + canonical encoding + identity hash stay public. Joint-mode attempts without admin unlock raise `ResonantError(code="carl.resonant.private_required")` — no pedagogical fallback (the math IS the commercial differentiator). `carl resonant eval` CLI surfaces this cleanly with exit code 2.
- **F4 (50da47f):** EML reward trained 7-param coefficients + init heuristics → `resonance/rewards/eml_weights.py`. Structure (depth-3 tree, composition rules, `score_from_trace` interface) stays public. `_random_init_weights()` fallback produces structurally sound but benchmark-inaccurate coefficients when admin gate is locked.
- **F5 (rolled into A-Day-4 ecd7174):** `admin.py::load_private` now tries `importlib.import_module(f"resonance.{module_name}")` first before the HuggingFace dataset fallback. Supports dotted submodule paths (`signals.constitutional`, `geometry.joint_cognize`, etc.) via the local fast path only.
- **F6 (this commit):** CI moat-boundary enforcement. `scripts/check_moat_boundary.py` walks every `.py` in `packages/carl-core/src/` + `src/carl_studio/` and parses ASTs; reports any module-level `import resonance` / `from resonance.X import Y` / `import terminals_runtime` / `from terminals_runtime.X import Y` as a violation. Lazy imports inside function bodies (the canonical admin-gate pattern) pass clean. GitHub Action at `.github/workflows/moat-boundary-check.yml` runs on every PR + push to main. 8 regression tests in `tests/test_moat_boundary.py` pin the behavior.

## Scope cut (with rationale)

- **Pre-commit hook config (pre-commit-hooks.yaml).** Python's `pre-commit` framework adds a dev-tooling dependency. Deferred to v0.18 — devs can run `python scripts/check_moat_boundary.py` manually, and CI catches everything before merge. Adding the pre-commit scaffold is a 10-line follow-up; skipped here to keep v0.17 focused.

- **AdminToken-verified runtime-resolver registration in private modules.** The seam exists (`Vault.register_runtime_resolver(kind, resolver, admin_token=token)` + `admin.issue_token()`), but no private module under `resonance/` currently uses it — F1/F2/F3/F4 are all `load_private(...)` lookups at call time, not resolver-chain registrations. The seam will activate when a real hardware-attested backend lands; for now it's regression-tested but dormant.

## Metrics

| Metric | Value |
|---|---:|
| Private LOC extracted to `resonance/` | ~1,166 (F1 556 + F2 ~300 + F3 ~80 + F4 ~230 incl. client stub + test scaffolding) |
| Public surface LOC delta | +~150 (client stubs + pedagogical fallbacks) |
| Moat-check script | 1 script + 1 workflow + 8 tests |
| Files changed in carl-studio | ~14 |
| Full-suite baseline | 3,707 pass, 3 skip, 0 fail (up from 3,599 pre-v0.17; +108 net new tests across Teams A + F) |
| Pyright strict errors on new/modified files | **0** |
| Ruff warnings on new/modified files | **0** |

## Blast radius (actual vs predicted)

| Extraction | Predicted LOC | Actual LOC | Callers updated |
|---|---:|---:|---:|
| F1 constitutional | 556 | 556 | 0 in production (fsm_ledger.py's imports stayed stable via facade) |
| F2 heartbeat | ~300 | ~300 (public facade 553, private 231 = 784 total, 448 of the original was pedagogical docstring + API; 300 of the pinned numerics moved) | 0 (`__init__.py` re-exports unchanged) |
| F3 joint cognize | ~80 | ~80 (private-side cognize_joint callable) + ~90 public-side delegate scaffolding | `cli/resonant.py` got 7 LOC to surface `ResonantError` on `carl resonant eval` |
| F4 EML reward weights | ~150 | numeric constants + init heuristic moved; structure preserved public | `training/rewards/eml.py` delegates via `admin.load_private('rewards.eml_weights')`; random-init fallback added |

## Decisions made

1. **Public wire format > moat when the two conflict.** F1's `LedgerBlock` + `encode_action_features` stay public because the TS npm sibling (`@terminals-tech/emlt-codec`) parity via `ledger_vectors.json` is a v0.17 §1.2 must-not-break contract. The moat-relevant part (genesis/append/sign) gates behind admin.
2. **Pedagogical fallbacks are per-primitive, not universal.** Heartbeat (F2) keeps a simple-reference fallback that satisfies the theorem's bounded-invariants witness — competitors can reproduce the paper's structural claims but not the benchmarked coefficients. Resonant joint-mode (F3) has NO fallback — the math is the whole point; without the private runtime, `joint`-mode Resonants raise `carl.resonant.private_required`. EML reward (F4) falls back to random-init when private unavailable — structurally sound, benchmark-inaccurate.
3. **`@pytest.mark.private` pattern.** Registered in `pyproject.toml` during F1. Any test that asserts against benchmarked numbers or private implementation details gets this marker; tests skip cleanly when the `resonance` package isn't on sys.path.
4. **Local resonance fast path (F5) over HF dataset.** Terminals-team machines have the private repo pip-installed editable at `/Users/.../models/resonance/`; `load_private` tries `importlib.import_module("resonance.<name>")` first. The HF dataset fallback remains for distributed access — `wheattoast11/carl-private` — but only handles flat module names. Dotted paths (`signals.constitutional`) require the local package; we refuse with a clear message rather than silent failure.
5. **AST-based moat check, not grep.** `grep` would false-positive on string literals like `"resonance"` or docstrings mentioning the package. The AST walker inspects only `Import` / `ImportFrom` nodes at module-body level, which is precise.

## Handoff

- **Teams B / C / D / E are unblocked.** Nothing in their scope touches moat files.
- **Future private features register via the existing seam.** Either `load_private(...)` lazy-import inside a function body (F1/F2/F3/F4 pattern) or `Vault.register_runtime_resolver` with an `AdminToken` (dormant seam, regression-tested).
- **Adding a new private module** to `resonance/`: drop it in the right subpackage (`signals/`, `rewards/`, `geometry/`, `ttt/`, `deployment/`), load from carl-core via `admin.load_private("<subpkg>.<name>")` inside a function. CI moat check stays green automatically.
- **Auditing moat compliance** post-v0.17: run `python scripts/check_moat_boundary.py` locally, or rely on the GH Action on every PR.

## Sign-off

Team F sprint complete. Public/private boundary is now enforced by CI — no more gradual leakage possible without an explicit PR that fails the check.
