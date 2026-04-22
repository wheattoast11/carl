---
last_updated: 2026-04-22
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.17.1-planned
classification: internal — CLI UX + dependency-probe hardening MECE plan
---

# v0.17.1 · CLI UX + Dependency-Probe Hardening

**Status:** plan (awaiting Tej sign-off) · **Trigger:** user hit
`ValueError: Unable to compare versions for huggingface-hub>=1.3.0,<2.0:
need=1.3.0 found=None` during first-run `carl init`; the traceback
propagated out of `_training_extras_installed()` and halted the wizard.

## 0. Context in one paragraph

`carl init` works flawlessly up to extras-check. Then `import transformers`
fires a `ValueError` (not `ImportError`) because `huggingface_hub`'s pip
metadata is corrupt on the local machine (two competing dist-info dirs).
The current probe catches only `ImportError`, so the exception hits the
outer wizard and kills the session. Two jobs: (a) harden every
optional-dep probe in Carl against this class of failure (auto-heal
when possible, degrade gracefully when not), and (b) cash in on the
moment to upgrade the CLI's interaction model across every menu so a
second user never sees a bare `typer.prompt("Pick one", default="1")`
again.

---

## 1. Root cause — established, not speculated

### 1.1 Evidence (captured live from Tej's machine)

```
huggingface_hub.__version__       = "1.9.2"        # import works
huggingface_hub.__file__          = /Users/terminals/miniforge3/lib/python3.12/site-packages/huggingface_hub/__init__.py

importlib.metadata.version("huggingface-hub")   → None
importlib.metadata.version("huggingface_hub")   → None

site-packages state:
  huggingface_hub                       (dir)
  huggingface_hub-1.5.0.dist-info       (dir, STALE + corrupted)
  huggingface_hub-1.9.2.dist-info       (dir, current)

pip show huggingface-hub  →  ValueError: invalid literal for int() with base 10: ''
                             (pip's own metadata reader can't parse METADATA)

transformers metadata version = "5.3.0.dev0"  (bleeding-edge dev build)
transformers import           → ValueError  ← raised from dependency_versions_check
```

### 1.2 The exact causal chain

1. User previously installed `huggingface_hub==1.5.0` via pip.
2. A subsequent upgrade (pip or conda) left the old `1.5.0.dist-info`
   directory on disk while also installing `1.9.2.dist-info`.
3. The `1.5.0.dist-info/METADATA` file is corrupt — its
   `Metadata-Version:` field parses to the empty string `""`.
4. `transformers 5.3.0.dev0` calls
   `importlib.metadata.version("huggingface-hub")` from
   `dependency_versions_check.py`.
5. `importlib.metadata` iterates dist-info dirs, hits the corrupted one
   first, fails to extract a version, and returns `None` (not
   `PackageNotFoundError`).
6. `transformers` treats `None` as a mismatch and raises `ValueError`.
7. `carl init._training_extras_installed()` only catches `ImportError`
   → the `ValueError` propagates out of `_offer_extras` → wizard dies.

### 1.3 Why this will happen again to other users

- Conda + pip mixing in the same env (the default miniforge pattern)
- `pip install --no-deps` / `--target` hacks in research setups
- Interrupted `pip upgrade` (Ctrl-C mid-install leaves dist-info stubs)
- Dev builds of transformers that pin exact huggingface-hub versions
- Any package the user upgrades outside pip (git clone, manual copy)

The class of failure — "import succeeds, metadata lookup returns None,
downstream package explodes" — is common enough that it needs first-class
treatment in Carl, not a bolt-on `except Exception` in one function.

---

## 2. Decision summary — recommendations up front

| Decision | Pick | One-line reason |
|---|---|---|
| **Immediate unblock** | `pip install --force-reinstall --no-deps huggingface_hub` + stale-dist-info sweep | Surgical. No side effects. |
| **Probe hardening** | Dedicated `carl_core.dependency_probe` module + typed result | Catches `Exception`, returns remediation hint; reusable across init / env / doctor. |
| **Auto-heal policy** | Offer `--force-reinstall --no-deps` with user consent, never silent | Safer than silent auto-repair; user retains agency. |
| **Selection library** | `questionary` (MIT, stable, tiny) behind a `cli/ui.py` facade | Industry-standard arrow-key UX; clean non-TTY fallback. |
| **Mistype recovery** | Arrow-key select (Enter commits); drop numeric auto-advance | Eliminates the mistype concern entirely. See §5.2. |
| **Form vs sequential** | Keep sequential for now; upgrade each prompt's internals first | Preserves familiarity; form view is v0.18. |
| **carl.camp auth UX** | gh-style: single arrow-key menu → default "sign in via browser" → auto-open + poll | Zero ambiguity, one Enter to sign in. |
| **Login-as-option on provider menu** | NO | Two concerns: provider = which model; auth = how you sign in. Keep separate. |
| **Integration surface** | Behind a version-gated flag (`CARL_CLI_UI=modern`) until v0.18 release; modern by default on new installs only | Tej is the only user right now; flip default on the tag that ships. |

---

## 3. Scope — MECE, by workstream

### 3.1 Track A · Immediate unblock (10 min, Tej runs once locally)

- **A1** Cleanup command (safe, verified):
  ```
  python -c "
  import shutil, pathlib
  p = pathlib.Path('/Users/terminals/miniforge3/lib/python3.12/site-packages/huggingface_hub-1.5.0.dist-info')
  if p.is_dir(): shutil.rmtree(p); print('removed', p)
  else: print('not present', p)
  "
  pip install --force-reinstall --no-deps huggingface_hub
  python -c "from importlib.metadata import version; print('OK:', version('huggingface-hub'))"
  python -c "import transformers; print('OK:', transformers.__version__)"
  ```
- **A2** Run `carl init --force` end-to-end; verify extras-check path
  succeeds with transformers already installed.

### 3.2 Track B · Dependency-probe hardening (code, ~300 LOC)

Files to create:
- `packages/carl-core/src/carl_core/dependency_probe.py`
  ```python
  @dataclass(frozen=True)
  class DepProbeResult:
      name: str                # PEP-503-normalized
      installed: bool
      version: str | None      # from importlib.metadata, NOT __version__
      import_ok: bool
      import_error: str | None
      metadata_error: str | None
      extras_marker: str | None  # "carl-studio[training]" etc.
      repair_command: str | None # "pip install --force-reinstall --no-deps X"

  def probe(package: str, *, import_name: str | None = None) -> DepProbeResult:
      """Classify optional-dep health: import-ok, metadata-ok, corrupt, or missing."""
  ```
- `packages/carl-core/tests/test_dependency_probe.py` — 6 cases:
  missing / import-ok-metadata-ok / import-ok-metadata-missing /
  import-ok-metadata-corrupt / import-fail-ImportError /
  import-fail-ValueError (the HF scenario, mocked).

Files to edit:
- `src/carl_studio/cli/init.py:362-368` —
  `_training_extras_installed()` becomes a thin wrapper over `probe()`
  and returns `DepProbeResult` instead of bool.
- `src/carl_studio/cli/init.py:371-410` — `_offer_extras()` gains a
  `_offer_auto_heal(probe_result)` branch that runs before the fresh
  install path when the probe reports `installed=True` but
  `metadata_error` set.
- `src/carl_studio/freshness.py` — new issue code
  `carl.freshness.dep_corrupt` (severity=error, category=environment,
  remediation from `DepProbeResult.repair_command`).
- `src/carl_studio/cli/startup.py::doctor` — surface `dep_corrupt`
  issues prominently (top of report, with one-line repair command).

Test additions:
- `tests/test_init_wizard.py` — mock a corrupt-metadata probe and
  assert `_offer_extras` surfaces the auto-heal prompt; assert
  wizard completes instead of dying.
- `tests/test_freshness_report.py` — assert new issue code.

**Blast radius:** ~12 files touched; 6 new tests; 3,729 → ~3,741 suite.

### 3.3 Track C · UX facade (code, ~250 LOC)

Files to create:
- `src/carl_studio/cli/ui.py` — single module that wraps questionary
  behind a stable facade. Public API:
  ```python
  def select(
      prompt: str,
      choices: list[Choice | str],      # Choice has (label, value, hint, badge)
      default: int = 0,                 # first-is-default by convention
      *,
      help: str | None = None,
      cancel_text: str = "Cancel",
      back_handler: Callable[[], None] | None = None,  # Ctrl-B if set
  ) -> str                              # returns Choice.value or ""
  def confirm(prompt: str, *, default: bool = True) -> bool
  def text(prompt: str, *, default: str = "", secret: bool = False) -> str
  def path(prompt: str, *, must_exist: bool = False, default: str = "") -> str
  ```
  Every call checks `sys.stdin.isatty()`; on False, routes to the
  `typer.prompt` legacy path so `echo "" | carl init` still works.
- `src/carl_studio/cli/ui_theme.py` — Rich-compatible theme tokens
  (`campColors` etc.); one place to change colors.
- `tests/test_cli_ui.py` — Click's `CliRunner` + questionary's testable
  `KeyInputs` helper; cover arrow-navigation, Enter-commits,
  Ctrl-C-cancels, non-TTY fallback path.

Files to edit (each migration is a one-line call-site swap):
- `src/carl_studio/cli/init.py` — provider menu (L244-249), camp-
  account confirm (L165), extras confirm (L383), project confirm
  (L431), sample-project confirm (L608), context confirm (L696, L708).
- `src/carl_studio/cli/platform.py` — preset menu (L185), tier menu
  (L236), reset confirm (L137), overwrite confirm (L162).
- `src/carl_studio/cli/project_data.py` — "Your pick" (L192), style
  (L200), all frees-text prompts L248–L266.
- `src/carl_studio/cli/env.py:22-36` — the central `_prompt_user`
  helper; single swap powers all 7 env questions.
- `src/carl_studio/cli/lab.py:188, 396, 646` — three confirms.
- `src/carl_studio/cli/startup.py:334` — project-init confirm.
- `src/carl_studio/cli/consent.py:80` — reset confirm.
- `src/carl_studio/cli/prompt.py:170, 176, 183` — three confirms +
  one password prompt; `secret=True` path in `ui.text`.
- `src/carl_studio/cli/chat.py:366` — `carl init` nudge confirm.

Dependencies added:
- `pyproject.toml [project.optional-dependencies]`:
  ```
  cli = ["questionary>=2.0,<3"]
  ```
  Pulled into `[all]` per the extras-coverage policy. Licenced MIT;
  ~100 KB; pure Python; no C deps. Lazy-imported inside `cli/ui.py`
  so `import carl_studio` stays light.

**Blast radius:** ~18 files touched; 1 new test module (~8 tests);
~250 net LOC.

### 3.4 Track D · carl.camp auth — gh-style (code, ~150 LOC)

Current flow (init.py:157–200):
1. "Already have one?" — **confirm**, default False
2. If no → `webbrowser.open("/auth/signup")` + bare `input()` to wait
3. Then `login_cmd()` runs separately (prompts again)

Three prompts, three Enters, confusing branching. Replace with:

1. `ui.select(
     "carl.camp account",
     [
       Choice("sign_in",       "Sign in with browser",    badge="recommended"),
       Choice("paste_token",   "Paste an auth token"),
       Choice("create_account","Create an account",       hint="opens carl.camp/auth/signup"),
       Choice("skip",          "Skip — configure later"),
     ],
     default=0,
   )`
2. "sign_in" → auto-open browser, start local OAuth callback listener
   on 127.0.0.1:<random> (same pattern as `gh auth login`), poll for
   token, time out at 120s with retry. On success: `c.ok("Signed in as X.")`
3. "paste_token" → `ui.text("Paste token", secret=True)` + validate
4. "create_account" → open signup URL, then auto-rerun step 1 with a
   short "Once you're done, press Enter to sign in." prompt
5. "skip" → persist intent; later commands re-surface as needed

Files to create:
- `src/carl_studio/auth/browser_flow.py` — local-callback OAuth client
  (aiohttp-free; use stdlib `http.server`). Backed by carl.camp's
  existing `/auth/cli-callback` endpoint (carl.camp agent to confirm
  — flag in §9 coordination).

Files to edit:
- `src/carl_studio/cli/init.py::_ensure_camp_account` — rewrite to
  use `ui.select` + the new browser flow.
- `src/carl_studio/cli/prompt.py:170, 176` — remove the two interim
  confirms; the ui.select replaces both.

**Blast radius:** ~5 files touched; 3 new tests (browser-flow unit,
mock OAuth server; select-branch integration).

### 3.5 Track E · Documentation + release (docs only)

- `docs/v17_cli_ux_doctrine.md` — new doc. Captures: (a) first-is-
  default rule, (b) arrow-key-select-over-numeric rule, (c) never
  auto-advance on single keypress rule, (d) non-TTY fallback
  contract, (e) `ui.select/confirm/text/path` as the only prompt
  API going forward, (f) gh-style browser-auth pattern.
- `docs/v17_dep_probe_doctrine.md` — new doc. Captures the
  import/metadata/corrupt taxonomy; mandates `dependency_probe.probe()`
  in every optional-dep check site.
- `CLAUDE.md` — new section "CLI UI doctrine (v0.17.1+)" pointing
  at the two docs; +3 lines about `cli/ui.py` being the single
  prompt surface.
- `CHANGELOG.md` — one block covering dep-probe + UI facade + camp
  auth flow.

---

## 4. Verification plan

### 4.1 Unit (fast, deterministic)

```
pytest packages/carl-core/tests/test_dependency_probe.py -q --tb=short
pytest tests/test_cli_ui.py -q --tb=short
pytest tests/test_init_wizard.py -q --tb=short
pytest tests/test_freshness_report.py -q --tb=short
```

Target: 4 new modules, ~20 new tests, 0 regressions, full suite ≤ +15%
time delta per §6 policy.

### 4.2 Static (per-file; no repo-wide lint)

```
pyright src/carl_studio/cli/ui.py src/carl_studio/cli/ui_theme.py \
        src/carl_studio/auth/browser_flow.py \
        packages/carl-core/src/carl_core/dependency_probe.py
ruff check <same files>
```

### 4.3 System integration (Tej runs, ~5 min)

```
# A. Confirm the HF corruption repaired and transformers loads clean
python -c "from carl_core.dependency_probe import probe; print(probe('huggingface_hub'))"
python -c "import transformers; print('ok', transformers.__version__)"

# B. Smoke the wizard end-to-end
carl init --force

# C. Smoke the doctor surface
carl doctor
```

### 4.4 User acceptance (interactive, ~10 min)

UAT script for Tej to walk through as an end user:

1. `carl init --force` — observe:
   - [ ] provider menu renders as arrow-key select, not numeric list
   - [ ] first option highlighted by default
   - [ ] up/down to switch, single Enter commits
   - [ ] Ctrl-C cancels cleanly
2. Deliberately cause extras corruption:
   ```
   # (run only on a scratch env)
   touch /path/to/site-packages/huggingface_hub-99.99.99.dist-info
   echo "Metadata-Version: " > .../METADATA
   ```
   Then `carl init --force` → observe:
   - [ ] auto-heal prompt appears
   - [ ] running the repair fixes the env in-place
   - [ ] wizard continues from the next step, no restart
3. Sign-in flow:
   - [ ] menu shows "Sign in with browser" as default
   - [ ] Enter opens browser + spins a local listener
   - [ ] signing in on the web returns focus to CLI
   - [ ] skip path exits cleanly
4. `carl env` — run through all 7 questions:
   - [ ] each question uses arrow-key select (or typed text where
     free-form)
   - [ ] --resume works if you Ctrl-C mid-flow
5. Non-TTY fallback — pipe: `echo "" | carl init --force`:
   - [ ] does NOT hang; writes `--help`-style diagnostic and exits 2

### 4.5 Regression guardrails

- `pytest -q --ignore=tests/test_uat_e2e.py --ignore=tests/test_uat.py`
  in CI; assert 3,729 pre-existing pass → 3,729 + ~20 new.
- `python scripts/check_moat_boundary.py` — new modules must not
  import `resonance.*` / `terminals_runtime.*`.

---

## 5. UX analysis — the full pros/cons table

### 5.1 Library choice

| Library | Arrow-key | Form view | Non-TTY | Pyright | Size | License | Last rel. | Recommend |
|---|---|---|---|---|---|---|---|---|
| `questionary` | ✅ | via checkbox | ✅ auto | ✅ | ~100 KB | MIT | active | **primary** |
| `prompt_toolkit` | ✅ | ✅ | ✅ | ⚠ looser | ~1.5 MB | BSD | active | too heavy |
| `rich.prompt` | ❌ | ❌ | partial | ✅ | already in | MIT | active | baseline only |
| `inquirer` | ✅ | ⚠ | ❌ hangs | ⚠ | ~200 KB | MIT | dormant | skip |
| `beaupy` | ✅ | limited | ✅ | ✅ | ~50 KB | MIT | active | alt; smaller |
| `simple-term-menu` | ✅ | ❌ | ❌ hangs | ⚠ | ~200 KB | MIT | stagnant | skip |

Primary: `questionary` — wraps `prompt_toolkit` but exposes a tiny surface
that matches our needs exactly and degrades cleanly on non-TTY. (If
Rich-native styling matters more than battle-testing, `beaupy` covers
90% of the API and is Rich-native by construction.)

**Pyright caveat:** questionary does NOT ship `py.typed`. We add the
package to `[tool.pyright].stubPath` with a 1-file stub, or set
`reportMissingTypeStubs = false` scoped to `cli/ui.py`. Precedent in-
repo: the httpx treatment for the x402 module.

**Back-navigation:** `ui.select` binds Ctrl-B as "go back" when the
caller supplies a `back_handler`. Matches poetry issue #8023 and the
gh pattern for device-code flows.

### 5.2 Mistype recovery — why arrow-key wins over numeric-auto-advance

| Pattern | Keystrokes for "pick item 3 of 5" | Recovery from mistype | Industry usage |
|---|---|---|---|
| Numeric + Enter | "3" + ↩ | Re-prompt | legacy |
| Numeric auto-advance | "3" | ❌ none — already committed | rare, user-hostile |
| Number-to-jump + Enter | "3" + ↩ | Up/Down to fix, Enter | questionary supports |
| Arrow + Enter | ↓↓ + ↩ | Up/Down to fix, Enter | **modern default** |

Arrow+Enter and Number-to-jump are **the same committed UX under
questionary**: pressing "3" jumps focus to item 3, but doesn't commit
— user still presses Enter. So you can type digits as shortcuts AND
recover via arrows AND commit with Enter. This is the same as:
`gh`, `poetry`, `cargo new`, npm `create-*`, Claude Code's own menus.
The user's "auto-submit on number" concern resolves to "you can type
the digit as a hotkey, nothing auto-commits until Enter." Zero extra
keystrokes vs numeric-auto-advance; full mistype recovery.

### 5.3 Tab-form vs sequential — when to prefer which

| Context | Sequential wins | Form wins |
|---|---|---|
| New user, first-run | ✅ (familiarity) | |
| Returning user, editing 1 field | | ✅ (no scroll) |
| Conditional logic (A depends on B) | ✅ (clarity) | ⚠ (state mgmt) |
| ≤ 4 questions total | ✅ (no visual weight) | |
| ≥ 6 questions total | | ✅ (single-screen context) |
| Review + correction pass | | ✅ |

**Recommendation for `carl env`'s 7 questions:**
- Phase 1 (this release): keep sequential, upgrade each question's
  internals via `ui.select`.
- Phase 2 (v0.18): add a **review screen** after question 7 that
  shows all answers as a form; user can tab to any field and re-edit.
- Phase 3 (v0.19+): optional full form view as alternative entry
  point (`carl env --form`).

This is the cheapest path that preserves the sequential flow for
first-timers while giving returning users the correction surface they
need.

### 5.4 Login-as-option on provider menu — rejected

User floated: fold "sign in with browser" onto the provider-selection
menu so it's one less decision point.

**Argument against:** these are orthogonal concerns.
- Provider selection = "which model do I talk to" (per project / per
  call)
- carl.camp auth = "am I signed in to the managed tier" (account-level)

Folding them confuses the mental model: if I pick "Anthropic + sign in
with browser", what does "sign in with browser" mean in the Anthropic
context? Anthropic doesn't use carl.camp's OAuth.

Recommendation: keep them separate. Both use `ui.select` so they feel
identical; each appears in its natural position in the wizard (account
→ provider → extras → project). Tej's real concern — "too many
Enters" — is solved by arrow-key select, not by merging menus.

---

## 6. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| `questionary` adds a dep that breaks on some terminal | Medium | Fallback to `typer.prompt` when TTY unavailable; integration test pipes `echo "" \| carl init` |
| Auto-heal silently modifies user's env | Medium | NEVER silent — always prompt `[Y/n]` before running `pip install --force-reinstall --no-deps`; log the command |
| Our `ui.select` disagrees with existing Rich theme | Low | `ui_theme.py` is the single palette; questionary accepts Rich colors directly |
| First-run wizard calls out to a remote OAuth that fails | Medium | Browser-flow times out at 120s; fallback "paste your token instead" branch of the select always present |
| Migration touches too many files in one PR | Medium | Track C splits by file; each `cli/*.py` migration is its own commit; reviewers can cherry-pick |
| Tests become flaky on terminal-shape assumptions | Low | `CliRunner` isolates; questionary has `KeyInputs` deterministic driver; no real TTY needed |

---

## 7. Non-goals (out of scope for v0.17.1)

- **Full Textual TUI for `carl env`.** Deferred to v0.18.
- **Multi-window layouts.** No splits, no status bars, no heartbeat
  dashboards in the wizard. Rich + questionary only.
- **Themed output (dark/light toggle, NO_COLOR beyond default).**
  Palette stays as-is; theme tokens live in `ui_theme.py` for a
  later pass.
- **Remote entitlement verification UX.** The carl.camp sign-in flow
  ends at "you are signed in." Tier-gated features re-prompt as
  needed.
- **Telemetry for menu interactions.** Privacy-first consent already
  controls this; opt-in additions are a v0.18+ privacy review.

---

## 8. Timeline (solo-speed, Tej + one agent)

| Day | Track | Deliverable |
|---|---|---|
| 0 | A | Unblock local machine (Tej runs §3.1 snippet) |
| 0 | B | `dependency_probe` module + tests; `_training_extras_installed` wrapped |
| 1 | B | freshness integration + doctor surface; UAT §4.3 |
| 1 | C | `cli/ui.py` + `ui_theme.py` + questionary extra + tests |
| 2 | C | init.py + platform.py + env.py migrations |
| 2 | C | project_data.py + lab.py + startup.py + consent.py + prompt.py + chat.py migrations |
| 3 | D | carl.camp auth flow + browser-flow tests |
| 3 | E | doctrine docs + CLAUDE.md + CHANGELOG |
| 3 | — | Full UAT §4.4 + ship v0.17.1 |

---

## 9. Cross-system coordination (carl.camp side)

One open question for the carl.camp agent before Track D ships:

- **Does carl.camp already expose a CLI-callback endpoint** (e.g.
  `POST /auth/cli-callback` that receives a local `127.0.0.1:port`
  and returns a short-lived token after browser-side SSO completes),
  or do we need to add one?

If not exposed, v0.17.1 ships Track D with **paste-token as the default
sign-in path** and arrow-key "sign in with browser" opens the account
page for manual copy/paste. We add the OAuth callback on the carl.camp
side in parallel; once live, we flip the default back to "browser auto-
flow" without a Carl release.

---

## 10. Critical files — quick reference

| Area | Path |
|---|---|
| Failing probe today | `src/carl_studio/cli/init.py:362-368` |
| Wizard root | `src/carl_studio/cli/init.py:36-121` |
| 7-question env wizard | `src/carl_studio/cli/env.py`, `env_setup/questions.py` |
| First-run nudge | `src/carl_studio/cli/chat.py:358-371` |
| Credential require() | `src/carl_studio/cli/prompt.py:143-190` |
| carl.camp login | `src/carl_studio/cli/platform.py` (login_cmd) |
| Doctor surface | `src/carl_studio/cli/startup.py::doctor` |
| Freshness model | `src/carl_studio/freshness.py` |
| Error doctrine | `packages/carl-core/src/carl_core/errors.py` |
| New: UI facade | `src/carl_studio/cli/ui.py` (to create) |
| New: dep probe | `packages/carl-core/src/carl_core/dependency_probe.py` (to create) |
| New: UX doctrine | `docs/v17_cli_ux_doctrine.md` (to create) |
| New: dep-probe doctrine | `docs/v17_dep_probe_doctrine.md` (to create) |

---

## 11. Sign-off checklist (when complete)

- [ ] Track A — user's machine unblocked; `carl init --force` reaches
      the project step cleanly
- [ ] Track B — 6 probe tests + wizard integration test; full-suite
      still green
- [ ] Track C — 18 file migrations + 8 ui tests; non-TTY fallback
      verified
- [ ] Track D — carl.camp OAuth flow working end-to-end (or paste-
      token fallback documented if coordination pending)
- [ ] Track E — two doctrine docs + CLAUDE.md block + CHANGELOG
- [ ] UAT §4.4 walked end-to-end; all checkboxes ticked
- [ ] `pyright` strict 0 / `ruff` 0 on all new + modified files
- [ ] `python scripts/check_moat_boundary.py` clean
