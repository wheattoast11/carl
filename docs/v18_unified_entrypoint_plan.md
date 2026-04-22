---
last_updated: 2026-04-22
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.18.0-planned
classification: internal — v0.18 unified entry-point strategic plan
---

# v0.18 — Unified Entry Point, Project Gate, Carl Persona

**Status:** plan (awaiting Tej sign-off) · **Companion briefs:** `docs/v17_cli_ux_and_dep_probe_plan.md` (v0.17.1 shipped), `carl-camp-agent-context` (parity brief, §13 answered in `docs/platform-parity-reply-2026-04-22.md`).

This release is the **unification release**. It resolves the
bifurcations that accumulated through v0.14–v0.17 (init-vs-start, bare-
cli-vs-typer-subcommand, adapter-vs-compute-backend) and prepares the
carl-studio surface for v0.19+'s agentic-as-default flip. It does NOT
ship the full TUI — that's v1.0. v0.18 is the skeleton that makes the
TUI arrival non-disruptive.

## 0. The manifold (how I'm reading the ask)

Motif jot, for provenance:

```
λ(carl) ≡ one binary, three entry modes, shared state
  bare carl                     → agentic REPL          (today: chat_cmd; v0.18: unchanged; v0.19: upgrade)
  carl "<prompt>"               → REPL pre-submit       (v0.18: add)
  carl <verb>                   → deterministic handler (today: works)
  carl -p "<prompt>" / --print  → SDK mode, no React    (v0.19+)

Project ≡ directory with .carl/                         // local state holder
  project_color = hue(sha256(name))                     // kubectl-context style
  theme persists in .carl/theme.json                    // Carl vs Carli already exists

Onboarding ≔ once(~/.carl/.initialized)
  greet(sprite)
    → ask(name)
    → choose(primary_mode: agent | manual)
    → if agent: provider + key
    → offer(carl.camp: sign_in | skip)
    → consent
  never re-shown

Slash command trichotomy (from claude-code leak):
  PromptCommand    — expands to hidden user message (/commit → "please commit")
  LocalCommand     — synchronous text output      (/clear, /help)
  LocalJSXCommand  — renders Rich component       (/config, /status)

Realizability(C) ≔ ∀s. predicate(s) → witness_recorded(InteractionChain)
  already: @coherence_gate, @tier_gate, @consent_gate
  post-v0.17 Team D: content-hashed audit chain
  carl.camp cap: POST /api/ledger/append (ed25519 countersig)
  NO new blockchain. The chain IS the contract.

Workspace (from OpenHands):
  LocalWorkspace | DockerWorkspace | RemoteAPIWorkspace
  ↔ compute_backend: local | runpod | hf_jobs | remote_api
  post-v0.17.1 our split unlocks this cleanly
```

The SOUL-adjacent frame — "singular coherent manifold across
terminals / agent-zero / Claude" — reads to me as: don't design a new
system; **extend what's already resonant**. The parity brief plus
InteractionChain plus coherence-gate plus the handle runtime already
form the skeleton. v0.18 is ribbons, not bones.

## 1. Scope — what this release IS (and isn't)

### IS

- A unified `carl` entry model: one binary, three modes, shared state.
- A project-context gate: `.carl/` directory is the world-root; ops that
  need a project fail fast with a clear "not in a carl project" message
  when called outside.
- A once-ever persona-led onboarding (sprite + greeting + name +
  primary-mode + provider + carl.camp + consent).
- A slash-command trichotomy (`PromptCommand` / `LocalCommand` /
  `LocalJSXCommand`) that unifies `carl flow` ops with REPL slashes.
- Session/run alignment with carl.camp migration 025 (sessions as
  first-class, runs as children).
- Consolidation of the v0.18 backlog surfaced by the Explore agent
  (14 items; one highest-risk 3x-deferred item flagged).

### IS NOT

- Full Textual TUI. Rich + questionary today; Textual when v1.0 ships.
- A new blockchain or tangle. The InteractionChain + carl.camp ledger
  is already the consensus layer.
- A rewrite of the agentic tool loop. `chat_agent.py` + `ToolDispatcher`
  stay; the REPL is refined, not replaced.
- Migration of the 94-command surface to namespace-isomorphism. That
  belongs to v0.19 after this foundation lands.
- Sprite art beyond 10–15 lines of ASCII. No Disney. Terminal-authentic.
- Messaging-gateway surfaces (Telegram / Slack / etc.). Out of scope
  per Hermes-digest avoidance note.

## 2. Decision log — Tej's specific asks answered

| Question | Recommendation | Rationale |
|---|---|---|
| **Unified entry vs bifurcation** | One binary: bare `carl` → agentic REPL; `carl <verb>` → deterministic; first-run hook gates onboarding once. Kill standalone `carl init` / `carl start` as mandatory gates. | Matches claude-code / opencode / hermes. No separate onboarding binaries. |
| **Interactive vs non-interactive** | TTY + env var `CARL_NON_INTERACTIVE=1` + `--json` + stdin-pipe detection gate modes at runtime. Not separate commands. | `ui.py` already does `_isatty()` — extend the rule to agentic REPL too. |
| **Persona sprite** | 12-line ASCII octopus, rendered once at onboarding + in `carl doctor --verbose`. Carl-methodical vs Carli-warm already exists in `project_data.py`; onboarding just picks one. | Keep the brand surface minimal. No sprite in every command — that's clutter. |
| **Onboarding shape** | Progressive disclosure. Sequential (arrow-key select via `cli/ui.py`). 6 steps. Each step is resumable if interrupted. `~/.carl/.initialized` is the marker. | Wizards beat forms for ≥6 fields with branching (per UX research in v0.17.1). |
| **Default primary mode** | Default to **manual** (today's CLI), with onboarding asking *once* if the user wants to make agent-mode the bare-`carl` default. Persisted to `~/.carl/config.yaml::default_entry = "repl" \| "help"`. | Matches user's "progressive disclosure" ask. Opt-in agentic flip respects the existing CLI muscle memory. |
| **Login-as-option on provider menu** | STILL NO. Auth and provider are orthogonal. Keep separate (per v0.17.1 decision). | Validated again. Folding them confuses mental model. |
| **Color-coded CLI per project** | Hash project name → hue, render in prompt + status lines. Persisted to `.carl/theme.json`. Optional. | Matches `kubectl` / Docker-compose contexts. |
| **Realizability / blockchain** | Use what we have: `InteractionChain` (post-v0.17 content-hashed) + carl.camp `/api/ledger/append` (ed25519 countersig). Do NOT introduce a new chain. Formalize the existing gates as a "realizability theorem" doc. | InteractionChain IS a merkle-ish append-only log. Blockchain-by-another-name without the coordination overhead. |
| **TUI rewrite now** | DEFER to v1.0. v0.18 adds the entry-point structure that makes the TUI non-breaking to ship later. | Scope discipline. Rich + questionary covers 90%. |
| **MCP / plugin surface** | Adopt claude-code's `mcp add <name> <command>` ergonomics; our existing `cli/lab.py::mcp_serve` covers the server side. Full plugin model is v1.0. | Matches gh/opencode/hermes. |

## 3. Architecture — the four primitives v0.18 ships

### 3.1 Entry-point router (`carl_studio/cli/entry.py`, new)

```python
def _route(argv: list[str]) -> Callable[[], None]:
    """Decide which handler to invoke based on argv + context.

    - Empty argv → agentic REPL (today: chat_cmd) OR first-run wizard if
      ~/.carl/.initialized missing.
    - First arg matches a registered subcommand → deterministic dispatch
      (today: typer app).
    - First arg is a non-command string → treat as REPL prompt
      (claude-code-style; today we do `carl "<prompt>"` → ask_cmd; make
      it equivalent to REPL pre-submit).
    - --print / -p flag → SDK mode (no React-equivalent; just stream
      text to stdout). v0.18: stub; v0.19: wire.
    """
```

Rules:
- The router is the ONLY place that chooses between agentic-REPL and
  deterministic-dispatch. Every `__main__.py` / `cli/apps.py` call
  funnels here.
- Sub-command registration stays as-is (Typer app). The router *composes*
  with it.
- The `first-run` hook fires *inside* the router before dispatching —
  so any entry path triggers onboarding on a fresh machine.

### 3.2 Project context gate (`carl_studio/project_context.py`, new)

```python
@dataclass(frozen=True)
class ProjectContext:
    root: Path              # absolute path to the directory holding .carl/
    name: str               # from carl.yaml::name
    color: str              # deterministic from hash(name)
    theme: str              # "carl" | "carli"
    session_id: str | None  # current session (None if no active session)

def current() -> ProjectContext | None:
    """Walk up from CWD looking for a .carl/ directory. None if not in a project."""

def require(cmd_name: str) -> ProjectContext:
    """Raise a clear error if not in a project. Used by commands that need one."""
```

- Ops that **require** a project context: `carl train`, `carl eval`,
  `carl run <task>`, `carl publish`, `carl ship`, `carl agent publish`,
  `carl resonant publish` (anything that mints artifacts).
- Ops that **don't** require a project: `carl doctor`, `carl update`,
  `carl camp login`, `carl config`, `carl init`, bare `carl` REPL
  (which can operate project-less and offer to bootstrap).
- Gate message: clear, one line, suggests `carl init` or `cd` to a
  project. No traceback.

### 3.3 Onboarding persona (`carl_studio/persona/`, new)

- `persona/sprite.py` — 12-line ASCII octopus, two variants (Carl /
  Carli). Rendered via `rich.Panel` with color from theme.
- `persona/greeting.py` — message templates keyed by persona + time of
  day (optional spice).
- `persona/onboarding.py` — the once-ever flow. Six steps, each a
  `ui.select` or `ui.text` call, each resumable from
  `~/.carl/onboarding_state.json` so Ctrl-C doesn't orphan.
- Onboarding DAG:

```
  sprite_greet
  ├── ask_name (ui.text)
  │     ↓
  ├── pick_primary_mode (ui.select: [manual=default, agentic])
  │     ↓
  ├── pick_provider (ui.select: [anthropic=default, openrouter, openai, skip])
  │     ↓
  │     if not skip: ui.text(secret=True, key_name)
  │     ↓
  ├── offer_camp (ui.select: [sign_in=default, paste_token, create_account, skip])
  │     ↓
  ├── consent (ui.confirm): observability off by default; prompt each
  │     ↓
  └── mark_done (~/.carl/.initialized; write ~/.carl/config.yaml)
```

### 3.4 Slash-command trichotomy (`carl_studio/cli/slash.py`, new)

Port the claude-code pattern, Python-idiomatic:

```python
class SlashCommand(Protocol):
    name: str
    help: str

class PromptCommand(SlashCommand):
    """Expands to a hidden user message in the REPL. Example: /commit."""
    def expand(self, args: str) -> str: ...

class LocalCommand(SlashCommand):
    """Synchronous text output. Example: /clear, /help."""
    def run(self, args: str) -> str: ...

class LocalJSXCommand(SlashCommand):
    """Renders a Rich component into the REPL scrollback. Example: /status."""
    def render(self, args: str, console: CampConsole) -> None: ...

# Registry
COMMANDS: dict[str, SlashCommand] = {
    "/status":   LocalJSXStatusCommand,
    "/train":    PromptTrainCommand,
    "/eval":     PromptEvalCommand,
    "/clear":    LocalClearCommand,
    "/help":     LocalHelpCommand,
    "/resume":   LocalJSXResumeCommand,
    "/config":   LocalJSXConfigCommand,
    "/mcp":      LocalJSXMcpCommand,
    ...
}
```

v0.18 ships the trichotomy + 6 core slashes (`/status`, `/train`,
`/eval`, `/clear`, `/help`, `/resume`). v0.19 migrates all `cli/flow.py`
ops into slashes.

## 4. MECE workstream split

Five tracks. Each has a single deliverable and a 2-line verification.

### Track A — Entry-point router (2–3 days)

**Deliverable:** `carl_studio/cli/entry.py` + wire `__main__` through
it. Bare `carl` hits first-run hook if unmarked; otherwise falls
through to today's `_default_to_chat`. `carl "<prompt>"` routes to
REPL with pre-submit.

**Verification:**
- `carl --version` still works (fast path bypasses router body).
- `carl "hello world"` opens REPL with the first user turn = "hello
  world".
- `carl train --config carl.yaml` still works (deterministic verb).
- First-run on a fresh machine (delete `~/.carl/.initialized`) triggers
  onboarding; subsequent runs skip it.

### Track B — Project-context gate (2 days)

**Deliverable:** `carl_studio/project_context.py` + adopt in the 6
project-requiring commands. `.carl/` directory is scaffolded by
`carl init` alongside `carl.yaml`. Prompt in REPL shows project name +
color + theme glyph.

**Verification:**
- `carl train` outside any project → clear error + exit 2 + suggestion.
- `carl train` inside a project → as today.
- `carl` REPL inside a project → prompt shows `[⊙ my-carl-project]` in
  the project hue.
- `carl session start` mints a session id and stores in
  `.carl/sessions/<id>.json`; prompt adds session short-id.

### Track C — Persona onboarding (2–3 days)

**Deliverable:** `carl_studio/persona/` + six-step onboarding DAG +
resume support. Existing `carl init` becomes a *thin wrapper* over the
DAG (kept for explicit invocation; first-run autolaunches it too).

**Verification:**
- Fresh machine: bare `carl` → sprite + greet + 6 steps → marker → REPL.
- Ctrl-C at step 3 → state saved → next invocation resumes at step 3.
- `~/.carl/onboarding_state.json` contains partial progress.
- User selects "manual" primary mode → next bare `carl` prints `carl --help`
  equivalent instead of opening REPL.

### Track D — Session/run abstraction + carl.camp parity (2 days)

**Deliverable:** `carl session start` / `carl session resume <id>` /
`carl session list`. Sessions map 1:1 to carl.camp migration 025 schema.
`carl run <task>` records a run inside the current session. Syncs via
authenticated HTTP to `POST /api/sessions/:id` (when the platform
exposes it — flagged open gap in parity brief §4.6).

**Verification:**
- `carl session start` returns an id; `.carl/sessions/<id>.json` exists.
- `carl run echo hello` records a run under the current session.
- `carl session list` shows all sessions with status + start time.
- Parity reply doc (§13) answers all five carl.camp questions.

### Track E — Slash-command trichotomy (2 days)

**Deliverable:** `cli/slash.py` + six core slashes + REPL dispatcher.
`carl flow "/train /eval"` still works via the same registry.

**Verification:**
- REPL `/help` shows all registered slashes with their kind badge.
- REPL `/train` expands to "please run carl train --config carl.yaml"
  as a user message (PromptCommand).
- REPL `/status` renders a Rich panel with session + project + run
  state (LocalJSXCommand).
- `/clear` clears REPL scrollback (LocalCommand).

## 5. Consolidated v0.18 backlog (from deferred-items sweep)

Ranked by value × (1/risk-of-further-deferral):

| Rank | Item | Source | Scope |
|---|---|---|---|
| 1 | Fix `hf_jobs` adapter/compute split | Tej's live report | ✅ DONE today (commit included below) |
| 2 | Unified entry-point router | Tej's ask + claude-code digest | v0.18 Track A |
| 3 | Project-context gate | Tej's ask | v0.18 Track B |
| 4 | Persona onboarding | Tej's ask + sprite idea | v0.18 Track C |
| 5 | Session/run abstraction + carl.camp parity | Parity brief §4.6 + §7 + §13 | v0.18 Track D |
| 6 | Slash-command trichotomy | claude-code digest | v0.18 Track E |
| 7 | Parity reply to carl.camp | Parity brief §13 | ✅ Writing today |
| 8 | Bulk `chain.record` → `@audited` migration (Teams B/C/E) | v0.17 Team D readout | v0.18 opportunistic during Tracks A–E touches |
| 9 | Unified scrubber registry (v0.17 D4) | v0.17 Team D readout | v0.18 opportunistic |
| 10 | Cross-WASM engine bit-identity validation | Deferred 3+ times | **HIGH RISK — force into v0.18 with a dated SLA** |
| 11 | Resonant `content_hash` ↔ `identity` reconciliation | v0.9 deferred §1.11 | v0.18 — carl.camp decision pending; flag in parity reply |
| 12 | Zombie-file sweeper in `carl doctor` | v0.9 deferred §1.12 | v0.18 — 20-line PR |
| 13 | `pre-commit-hooks.yaml` for moat-boundary check | v0.17 Team F | v0.18 polish |
| 14 | Full Textual TUI for `carl env` | v0.17.1 plan §5.3 | v0.19+ (not this release) |
| 15 | `carl env --form` alternative entry | v0.17.1 plan §5.3 | v0.19+ |
| 16 | Session API elevation to `carl_studio.__init__` | v0.17 Team A | v0.18 — bundled with Track D |
| 17 | `HandleRuntimeBundle` deprecation | v0.17 Team A | v0.18 — Teams B/C migrate |
| 18 | `carl resonant fit / compose` | v0.9 deferred §1.1 | v0.19 (needs admin-gated backend) |
| 19 | `carl marketplace buy` x402 | v0.9 deferred §1.1 | v0.19 (platform dep) |
| 20 | py2bend policy-head compile (HVM) | CLAUDE.md v0.17 | v0.20+ |
| 21 | Remote entitlement verification | CLAUDE.md v0.10 | v0.19 |
| 22 | AXON HTTP event forwarder | CLAUDE.md v0.16 | v0.19 |
| 23 | Async-first HTTP rewrite | v0.17 arch plan | v0.19+ ("when swarm lands") |
| 24 | Five large-module decompositions | v0.17 arch plan | v0.19 refactor sprint |
| 25 | Bulk CLI reorg (94 commands namespace-iso) | v0.17 arch plan | v0.19 |

**Rule for Track F (deferred-items):** every item ranked 2–13 MUST
either ship in v0.18 OR get a dated SLA in v0.19. Items deferred 3+
times (ranked 10) get an owner assigned in THIS plan.

- **#10 (WASM bit-identity)** — assigned owner: the next available
  session. Due by v0.19 tag. If it slips again, we rename the harness
  to `tests/cross_engine_bit_identity_SLIPPAGE.rs` as a public
  shame-commit.

## 6. Answering the parity brief §13

See companion doc `docs/platform-parity-reply-2026-04-22.md`.
Summary: 1 already-correct, 3 gaps to fix in v0.18 Track D, 1 open
design question kicked back to carl.camp.

## 7. Realizability gating — the formalization (not a rewrite)

We DON'T add a blockchain. We formalize what we have.

**Theorem (realizability).** A gated operation `Op` with gate `G` is
realized iff:
1. `G.predicate(state)` evaluates to true,
2. the evaluation result is recorded as a `Step` on an
   `InteractionChain`, and
3. the resulting `Step.content_hash` is consistent (verifiable per
   v0.17 Team D).

**Composition.** Gates compose via AND (all-must-pass) and OR
(any-may-pass). Composite witness = merkle root of constituent step
hashes. Already implicit in `InteractionChain`; codify in
`carl_studio/gating.py::CompositeGate`.

**Cross-actor.** When multiple users contribute to a shared artifact
(resonant / agent card), the carl.camp `constitutional_ledger_blocks`
table aggregates per-user chain heads into an ed25519-signed block —
that's the consensus layer. One-way CLI→platform push remains the
invariant (parity brief §2 Invariant 1).

**What's NOT needed.** A separate blockchain. A new consensus protocol.
An IOTA-tangle-like DAG on the CLI side. These would duplicate state
already held in the two constitutional surfaces (local chain + platform
ledger) and violate parity brief Invariant 3 ("shared codec, nothing
more").

**Deliverable.** `docs/v18_realizability_theorem.md` — the formal
writeup. 2–4 pages. Target audience: paper-reading agents. Not a
dependency for shipping the rest of v0.18.

## 8. Timeline (solo-speed)

| Day | Track | Deliverable |
|---|---|---|
| 0 | — | ✅ hf_jobs fix + v0.18 plan + parity reply (TODAY) |
| 1 | A | Entry-point router + tests |
| 2 | B | Project-context gate + REPL prompt upgrades |
| 3 | C | Persona + sprite + onboarding DAG |
| 4 | D | Sessions + parity §13 reply merge |
| 5 | E | Slash-command trichotomy + 6 core slashes |
| 6 | F | Opportunistic backlog pickups (items 8–13) |
| 6 | — | UAT, ship v0.18 |

## 9. Verification (overall)

- Full-suite ≥ 3746 (today's v0.17.1 baseline) + the new tests from
  each track. 0 regressions.
- `pyright` strict: 0 errors on new files.
- Ruff: 0 on new + migrated files.
- `carl` on a fresh machine (delete `~/.carl/`) runs onboarding end-to-
  end in <2 min without crashing.
- `carl train --config carl.yaml` in a v0.17.x project **still works**
  (backward compat through the legacy-backend fall-through). The
  v0.17.1 adapter fix (landed in §1) unblocks this.
- Parity reply committed to `docs/platform-parity-reply-2026-04-22.md`
  so carl.camp's agent can read + acknowledge.

## 10. Non-goals (explicit)

- **No Textual TUI.** Rich + questionary this release. Textual is v1.0.
- **No new blockchain.** InteractionChain + carl.camp ledger is it.
- **No messaging gateway.** Not Telegram, not Slack, not Signal. Stay
  focused on the RL-training surface.
- **No ML-classifier permissions.** We use `@coherence_gate` and
  `@tier_gate` because they're principled.
- **No public `.map` leak.** Every `admin.py`-gated thing stays in
  `terminals-runtime`.
- **No mandatory re-onboarding.** Existing users with
  `~/.carl/.initialized` present skip the sprite flow. New prompts
  only on schema-bump.
- **No Disney sprite.** 12 lines of ASCII. Two variants. Terminal-
  authentic.

## 11. Sign-off checklist (when v0.18 complete)

- [ ] Track A — entry-point router + first-run hook.
- [ ] Track B — project-context gate live on 6 ops.
- [ ] Track C — persona onboarding reaches all 6 steps; Ctrl-C
      resumable.
- [ ] Track D — sessions + runs persisted + sync to carl.camp (once
      the platform exposes `POST /api/sessions/:id` publicly).
- [ ] Track E — slash trichotomy + 6 slashes live in REPL.
- [ ] Track F — opportunistic backlog picks committed.
- [ ] Parity reply merged.
- [ ] Full suite ≥ v0.17.1 baseline + new tests; 0 regressions.
- [ ] `carl` fresh-machine UAT walked end-to-end.
- [ ] CHANGELOG entry.
- [ ] `docs/v18_realizability_theorem.md` drafted (optional polish).

## 12. Research provenance

- **claude-code leak (2026-03-31, v2.1.88 `.map` bundle)** — patterns
  lifted: entry-point enum, slash-command trichotomy, session JSONL
  per URL-encoded CWD, OAuth local-callback, 6-tier context compaction
  (partial — we don't need the full ladder).
- **opencode (`sst/opencode`)** — validated: `~/.<name>/` convention,
  `AGENTS.md` at repo root, merged global + project config, headless-
  serve pattern for future cloud attach.
- **OpenHands (All-Hands-AI)** — lifted: 3-tier workspace abstraction
  (Local/Docker/RemoteAPI) maps onto our compute_backend split.
  SDK-first shape validates our `chat_agent.py` + `ToolDispatcher`.
- **Hermes (NousResearch)** — closest philosophical competitor; lifted:
  `profile` isolation pattern (via `HERMES_HOME`), credential-pool
  design for managed-tier multi-key rotation, skills-as-procedural-
  memory as reference for Resonant lifecycle. Avoided: messaging
  gateway surfaces, `MEMORY.md`/Honcho dialectic model.
- **carl.camp parity brief (2026-04-22)** — consumed: MECE §1 surface
  split, API contract registry §4, schema traps §5, freemium split
  §6, mode system §7, idempotency §8, §13 question list.

## Sources

- [leaked Claude Code source mirror (yasasbanukaofficial)](https://github.com/yasasbanukaofficial/claude-code)
- [Reverse-Engineering Claude Code (sathwick.xyz)](https://sathwick.xyz/blog/claude-code.html)
- [How Claude Code Actually Works (Karan Prasad)](https://karanprasad.com/blog/how-claude-code-actually-works-reverse-engineering-512k-lines)
- [sst/opencode](https://github.com/sst/opencode)
- [All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- [hermes-agent CLI reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)
- [OpenHands SDK architecture](https://docs.openhands.dev/sdk/arch/workspace.md)
