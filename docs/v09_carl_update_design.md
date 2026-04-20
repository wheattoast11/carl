---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9-preview
status: roadmap
---

# Carl-Update: Self-Updating Agentic Meta-Pipeline — v0.9 Design

**Status:** Design document (no implementation yet)  
**Version:** v0.8.0 → v0.9-preview  
**Author(s):** Claude Code (agentic design)  
**Date:** 2026-04-20

## 1. Behavior Specification

### 1.1 Command Surface

`carl update` is a Typer command (registered via `wiring.py` alongside `doctor`, `init`, `start`) that surfaces recent knowledge changes to the user without applying them. It never auto-runs any modifications; its role is *agentic awareness* — surfacing deltas, not committing them.

```bash
# Explicit invocation
carl update                           # Run a full update check
carl update --summary-only            # Print 1-pager, exit
carl update --detailed                # Render full blast-radius breakdown
carl update --reset-interval          # Clear the 3-day timer and run immediately
carl update --dry-run                 # Surface deltas without DB persistence
```

### 1.2 Startup Behavior (3-day Nudge)

When a user runs any `carl` command (bare `carl`, `carl doctor`, `carl train`, etc.), if **>3 days** have elapsed since `carl.update.last_run_at` in `LocalDB.config`, a low-friction FYI fires **in the startup chain**:

```
  [camp.secondary]Insight: 3+ days of changes available[/]
  Latest carl-studio: v0.8.0 → v0.8.1 (3 new features, 1 security fix)
  Run: carl update   (or pass --skip-nudge to ignore)
```

**Key properties:**
- **Non-blocking, non-modal** — printed as a side notification (parallel to `carl doctor` freshness output).
- **Dismissable via `--skip-nudge`** flag in the main `carl` callback.
- **No prompts** — fully passive; user chooses to invoke `carl update` explicitly.
- **Respects consent:** only fires if telemetry consent allows network egress.

### 1.3 Full Update Report Output

When `carl update` runs, it renders a multi-panel summary:

```
  CARL Update

  [camp.primary]Knowledge Refresh[/]
  ✓ Last check: 2026-04-17 14:32Z  |  carl-studio v0.8.0  |  HEAD 6c46172

  [camp.primary]Recent Commits[/]
  3 commits (7 days)
    • config_registry: Typed config-key store (#42) — unlocks safer ~/.carl/ state migrations
    • freshness: Add CVE-scan integration (#41)
    • cli/startup: Reduce nudge noise (#40)

  [camp.primary]Dependency Deltas[/]
  transformers    5.2.0 → 5.3.0  [minor]  — Flash-3 attention in core
  peft            0.15.0 → 0.15.1 [patch]  — Bug fix: lora checkpoint loading
  anthropic       0.95.0 → 0.96.0 [minor]  — Batch API support (NEW)

  [camp.primary]Security Findings[/]
  ✓ No CVEs (0 issues in carl-studio + pinned deps)

  [camp.primary]What This Unlocks[/]
  • config_registry: Safe config migrations eliminate ~/.carl state drift
  • Flash-3 attention: Train 25% faster with transformers 5.3+
  • Batch API: Reduce claude API costs by 50% in agentic loops
```

## 2. Files to Create/Modify

### 2.1 New Files

#### `src/carl_studio/cli/update.py` (new)
- Typer command entry point: `@app.command(name="update")`
- Options: `--summary-only`, `--detailed`, `--reset-interval`, `--dry-run`, `--skip-nudge`
- Imports and coordinates `UpdateReport` construction via the `update/` package
- Renders via `CampConsole`

#### `src/carl_studio/update/` (new package)
Core internals; each module is a focused scan function:

- **`__init__.py`** — exports `UpdateReport`, `BlastRadiusEntry`, run functions
- **`git_scan.py`** — scans carl-studio + sibling repos (git log, tags, commits)
  - Returns: list of commits (hash, message, timestamp, author) since last check
  - Gated by telemetry consent (no egress, local-only git history)
- **`dep_scan.py`** — PyPI JSON API calls for installed packages
  - Queries `https://pypi.org/pypi/<pkg>/json` for latest version
  - Compares installed vs latest; categorizes by semver (major/minor/patch)
  - Returns: dict pkg → {installed, latest, severity}
- **`cve_scan.py`** — `pip-audit` wrapper or lightweight security check
  - Runs `pip-audit` if available, else logs "pip-audit not installed"
  - Returns: list of (cve_id, pkg, installed_ver, advisory)
- **`blast_radius.py`** — synthesizes positive impact framing
  - Takes git_delta + dep_delta + cve_findings
  - For each feature/dependency, emits `BlastRadiusEntry` with `direction: positive | breaking | neutral`
  - Crucially: **every entry includes `impact: str`** phrased as improvement unlock
  - Examples: "ConfigRegistry → typed config-key store unlocks safer ~/.carl/ state migrations"
- **`report.py`** — `UpdateReport` Pydantic model + rendering helpers
  - Analogous to `FreshnessReport`; uses same error code namespace

### 2.2 Modified Files

#### `src/carl_studio/cli/wiring.py`
- Register `update` sub-app at top level: `app.command(name="update")(update_cmd)`
- Pattern: identical to existing `init_cmd`, `flow_cmd` registration
- Optional-dependency stub for future: if `update` deps unavailable, fall back to stub (unlikely; deps already in tree)

#### `src/carl_studio/cli/startup.py`
- Modify `doctor()` callback or introduce a new `_check_update_nudge()` helper called from main `app.callback()`
- Gate on `needs_update_check()` and `consent_gate("telemetry")`
- Print nudge only if 3 days elapsed; respect `--skip-nudge`

#### `docs/operations.md` (or `CONFIGURATION.md`)
- Document env vars:
  - `CARL_UPDATE_CHECK_INTERVAL_DAYS` (default: 3)
  - `CARL_UPDATE_AUTO_RUN` (default: false; note that auto-run is NOT implemented — this is for future compat)
  - `CARL_UPDATE_SKIP_CVE` (skip `pip-audit` if set; useful in restricted envs)

## 3. Data Model

### 3.1 `UpdateReport` (Pydantic)

Mirrors `FreshnessReport` structure; all error codes under `carl.update.*` namespace.

```python
class BlastRadiusEntry(BaseModel):
    """A single improvement or breaking change."""
    code: str  # e.g., "carl.update.new_feature", "carl.update.breaking_change"
    code_path: str  # source (git commit, dep version, cve_id)
    category: Literal["feature", "bugfix", "security", "dependency"]
    direction: Literal["positive", "breaking", "neutral"]
    summary: str  # short 1-line title
    impact: str  # phrased as improvement unlock (e.g., "unlocks safer migrations")
    remediation: str | None  # action if breaking (e.g., "update dep X to >=Y.Z")

class UpdateReport(BaseModel):
    """Results of an update check."""
    last_check_at: datetime
    checked_at: datetime
    
    # Deltas
    git_delta: list[GitCommit] = Field(default_factory=list)
    dep_delta: dict[str, DepVersion] = Field(default_factory=dict)
    cve_findings: list[CVEFinding] = Field(default_factory=list)
    blast_radius: list[BlastRadiusEntry] = Field(default_factory=list)
    
    # Derived views
    has_breaking: bool = False
    has_security: bool = False
    summary: str = ""
```

### 3.2 Error Codes (under `carl.update.*`)

```python
CODE_NO_GIT_HISTORY = "carl.update.no_git_history"
CODE_DEP_UPDATE_AVAILABLE = "carl.update.dep_update_available"
CODE_DEP_MAJOR_UPDATE = "carl.update.dep_major_update"
CODE_CVE_FOUND = "carl.update.cve_found"
CODE_BREAKING_CHANGE = "carl.update.breaking_change"
CODE_NEW_FEATURE = "carl.update.new_feature"
CODE_NETWORK_ERROR = "carl.update.network_error"  # when PyPI call fails
```

## 4. Reuse Map

| Primitive | Usage | Rationale |
|-----------|-------|-----------|
| `FreshnessReport` | **Don't reuse** — `UpdateReport` is structurally different (focus on deltas, not freshness). Both live in `freshness.py` and `update/report.py` respectively. | Keeps concerns separate; each domain has its own report type. |
| `CARLSettings` | Read-only in `dep_scan.py` to detect user env (anthropic key, HF token for changelog context). | Already a dependency; no new imports needed. |
| `LocalDB.config_registry()` | Store `update.last_check_at` via `db.set_config("carl.update.last_check_at", timestamp_iso)` | Same pattern as freshness; typed, scoped keys in config table. |
| `LocalDB.get_config()` | Retrieve last check time; gate the 3-day nudge in startup. | Pre-existing; no new logic needed. |
| `BreakAndRetryStrategy` (carl-core) | Wrap PyPI JSON API calls in `dep_scan.py` with `retry(fn, policy=RetryPolicy(...))`. | Carl-core already provides; reduces boilerplate. |
| `emit_gate_event()` (if available) | Record update check as a telemetry event (Anthropic-internal observability). | Follows carl_studio.interaction pattern; optional (graceful no-op if unavailable). |
| `CampConsole` | Render output via `c.ok()`, `c.warn()`, `c.print()` in update.py. | Existing rendering toolkit; consistent UI. |
| `consent_gate("telemetry")` | Gate network calls (PyPI, CVE scan) on user consent. | Pre-existing consent layer; respects user privacy. |

**NEW primitives:** None. Existing tree provides all required building blocks.

## 5. Network Layer

### 5.1 PyPI JSON API

Each package in `dep_scan.py`:

```python
import urllib.request
import json

url = f"https://pypi.org/pypi/{pkg}/json"
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        data = json.loads(response.read().decode())
    latest = data["info"]["version"]
except (urllib.error.URLError, json.JSONDecodeError, KeyError):
    # Network error or malformed response — log and skip this pkg
    continue
```

- **No new dependency** — `urllib.request` is stdlib; `json` is stdlib.
- **Retry wrapper:** wrap entire call in `retry(fn, policy=RetryPolicy(max_attempts=3))`.
- **Consent check:** before any network egress, require `consent_gate("telemetry")` to pass.

### 5.2 CVE Scanning

- **Primary:** invoke `pip-audit` subprocess if installed (`shutil.which("pip-audit")`).
- **Fallback:** if not installed, log a soft warning and return empty findings.
- **No API calls** — `pip-audit` is local-only; it reads installed packages and checks a local database or cached copy.

## 6. Positive Framing

Every `BlastRadiusEntry.impact` MUST frame the improvement as an unlock, not a feature list.

**Bad (feature-list framing):**
- "Transformers 5.3 adds Flash-3 attention"

**Good (improvement-unlock framing):**
- "Flash-3 attention in transformers 5.3 unlocks 25% faster training"
- "ConfigRegistry adds typed config keys, unlocking safer ~/.carl/ state migrations"
- "Anthropic Batch API support unlocks 50% cost reduction in agentic loops"

**Pattern:** `[subject] unlocks [concrete user benefit]`

Each entry also includes `remediation: str | None` only when breaking (e.g., "update peft to >=0.15.1 to fix lora checkpoint loading").

## 7. Testing Shape (Sketch Only)

No implementation; list the 5–8 most important test cases:

1. **test_git_scan_finds_commits**
   - Fixture: temp git repo with 5 commits
   - Assert: `git_scan()` returns list of 5 commits with correct hashes/messages
   - Stub: subprocess mock for `git log`

2. **test_dep_scan_compares_versions**
   - Fixture: installed `transformers==5.2.0`, mock PyPI JSON endpoint with `5.3.0`
   - Assert: `dep_delta["transformers"] == {installed: "5.2.0", latest: "5.3.0", severity: "minor"}`
   - Stub: `urllib.request.urlopen` mock

3. **test_cve_scan_when_pip_audit_missing**
   - Fixture: `pip-audit` not in PATH
   - Assert: `cve_findings` is empty list; no exception raised
   - Stub: `shutil.which` returns `None`

4. **test_blast_radius_frames_positive**
   - Fixture: git commit "Add Batch API support", dep delta anthropic 0.95→0.96
   - Assert: `BlastRadiusEntry.impact` includes "unlocks 50% cost reduction"
   - Stub: none; pure logic test

5. **test_startup_nudge_when_3_days_elapsed**
   - Fixture: `last_check_at` is 4 days ago in `LocalDB.config`
   - Assert: `_check_update_nudge()` prints nudge to console
   - Stub: `CampConsole.info` mock

6. **test_startup_nudge_respects_skip_flag**
   - Fixture: `--skip-nudge` passed to `carl doctor`
   - Assert: nudge is NOT printed
   - Stub: typer context mock

7. **test_consent_gate_blocks_network**
   - Fixture: `consent_gate("telemetry")` returns `False`
   - Assert: `dep_scan()` and `cve_scan()` make no network calls
   - Stub: consent mock

8. **test_update_report_json_serializable**
   - Fixture: complete `UpdateReport` with commits, deps, cves, blast_radius
   - Assert: `json.dumps(report.dict())` succeeds; all datetime/object fields serialize
   - Stub: none; pure Pydantic test

## 8. Ship Sequencing (v0.9-preview)

### Phase 1: Foundation (Week 1)
1. Merge `src/carl_studio/update/` package (git_scan, dep_scan, cve_scan, blast_radius, report).
2. Merge `src/carl_studio/cli/update.py` (command entry point, basic rendering).
3. Add `CARL_UPDATE_CHECK_INTERVAL_DAYS` to `operations.md`.

### Phase 2: Integration (Week 2)
1. Wire `update` sub-app in `wiring.py`.
2. Add 3-day nudge to `startup.py` callback; gate on consent.
3. Add `--skip-nudge` flag to main `app.callback()`.
4. Add tests (shape from section 7).

### Phase 3: Polish (Week 3)
1. Render blast_radius with rich tables and color-coding (positive = green, breaking = red).
2. Integrate `emit_gate_event()` for telemetry (optional; can be added post-ship).
3. Final docs: add detailed examples to `operations.md` and a new `UPDATING.md` guide.
4. Tag v0.9-preview; no test regressions vs v0.8.0.

### Backward Compatibility
- v0.8.0 tests should pass unchanged; `update` is opt-in, never auto-applied.
- No CLI surface changes to existing commands; only new `carl update` verb and optional startup nudge.
- LocalDB schema unchanged; update state is stored as `config` rows (existing table).

## 9. Appendix: Example Output

### Full Report

```
────────────────────────────────────────────────────────────────────────
  CARL Update
────────────────────────────────────────────────────────────────────────

  ✓ Status: READY
  Last check: 2026-04-17  |  carl-studio v0.8.0  |  HEAD 6c46172

  ────────────────────────────────────────────────────────────────────
  Recent Commits (7 days, 3 new)
  ────────────────────────────────────────────────────────────────────

    6c46172  config_registry: Typed config-key store (#42)
    d4f8e2c  freshness: Add CVE-scan integration (#41)
    a9b1c3e  cli/startup: Reduce nudge noise (#40)

  ────────────────────────────────────────────────────────────────────
  Dependency Deltas (5 packages checked, 3 updates available)
  ────────────────────────────────────────────────────────────────────

    transformers       5.2.0 → 5.3.0  [minor]  Flash-3 attention
    peft               0.15.0 → 0.15.1 [patch]  Lora checkpoint fix
    anthropic          0.95.0 → 0.96.0 [minor]  Batch API support

  ────────────────────────────────────────────────────────────────────
  Security (CVE Scan)
  ────────────────────────────────────────────────────────────────────

    ✓ No CVEs in carl-studio + pinned dependencies (0 issues)

  ────────────────────────────────────────────────────────────────────
  What This Unlocks
  ────────────────────────────────────────────────────────────────────

    [positive] ConfigRegistry (v0.8)
      Typed config-key store unlocks safer ~/.carl/ state migrations
      Eliminates brittle string keys; future-proof config layer

    [positive] Flash-3 Attention (transformers 5.3)
      25% faster training with memory-efficient attention mechanism
      Backward-compatible; no code changes required

    [positive] Batch API (anthropic 0.96)
      50% cost reduction in agentic loops; async batch processing
      New feature; optional; recommended for production agents

    [bugfix] Lora Checkpoint Loading (peft 0.15.1)
      Fixes crash when resuming training from lora checkpoints
      Safe upgrade; no breaking changes

  ────────────────────────────────────────────────────────────────────

  Next: pip install --upgrade transformers peft anthropic
        OR: review the changelog at https://github.com/wheattoast11/carl
```

### Startup Nudge

```
  [camp.secondary]Insight: 3+ days of changes available[/]
  Latest carl-studio: v0.8.0 → v0.8.1 (3 new features, 1 security fix)
  Run: carl update   (or pass --skip-nudge to ignore)
```

---

**Design complete. Ready for v0.9-preview implementation.**
