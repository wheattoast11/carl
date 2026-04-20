# carl-studio operations reference

Surfaces that matter when you deploy carl-studio in a long-running context
(daemon, container, researcher's cluster, enterprise evaluator). Every
tunable knob is here; the source of truth in code is cited next to it.

## Quick orientation

- **Config hierarchy (CARLSettings):** env var > `~/.carl/config.yaml` > project-local `carl.yaml` > defaults. Resolution in `settings.py` (`CARLSettings.load`).
- **Settings env prefix:** `CARL_` via `SettingsConfigDict(env_prefix="CARL_")`. Any `CARLSettings` field is writable as an env var (e.g. `CARL_TIER`, `CARL_LOG_LEVEL`, `CARL_DEFAULT_MODEL`, `CARL_HUB_NAMESPACE`). Case-insensitive; nested via `__` delimiter.
- **Logging:** structured via `carl_studio.logging_config.configure_logging`. Idempotent, stderr-only, honors `CARL_LOG_LEVEL` + `CARL_LOG_JSON`.
- **State root:** `~/.carl/` — SQLite (`carl.db`), sessions (`sessions/`), wallet (`wallet.enc`), LLM cache (`llm_cache.db`), adapters, environments, keys. Override via `CARL_HOME` is partially honored — see the env var table below.
- **Env-var parser:** every daemon knob converges on `carl_studio.envutil.env_int / env_float / env_bool`. Malformed values never crash; they log `WARNING` and fall back to the default.

## Environment variables

### Credentials

| var | consumer | purpose |
|-----|----------|---------|
| `ANTHROPIC_API_KEY` | `llm.py`, `chat_agent.py`, `settings.py`, `cli/observe.py`, `coherence_observer.py` | primary chat + tools + observer |
| `HF_TOKEN` | `settings.py`, `tier.py`, `eval/runner.py`, `training/{trainer,pipeline}.py`, `compute/hf_jobs.py`, `bundler.py`, `environments/registry.py` | HF Hub auth. `huggingface_hub.get_token()` is consulted first; `HF_TOKEN` is the fallback. |
| `OPENROUTER_API_KEY` | `llm.py`, `cli/init.py`, `settings.py` | OpenAI-compatible router fallback (`https://openrouter.ai/api/v1`) |
| `OPENAI_API_KEY` | `llm.py`, `cli/init.py`, `settings.py` | OpenAI-compatible fallback |
| `OPENREWARD_API_KEY` | `environments/openreward.py` | OpenReward managed reward-model client (PAID-tier only) |
| `CARL_WALLET_PASSPHRASE` | `wallet_store.py` | Unlocks `~/.carl/wallet.enc`. Keyring is preferred when installed; env var is the fallback. Never logged. |
| `CARL_ADMIN_SECRET` | `admin.py` | Double-lock for private-runtime unlock (paired with hardware fingerprint). No-op when unset. |
| `CARL_LLM_API_KEY` | `llm.py` | Paired with `CARL_LLM_BASE_URL`. Explicit override; highest priority in provider auto-detect. |

### Logging + observability

| var | default | purpose |
|-----|---------|---------|
| `CARL_LOG_LEVEL` | `info` | Root logger level (`debug`/`info`/`warning`/`error`/`critical`). Unknown values coerce to `info`. (`logging_config.py`) |
| `CARL_LOG_JSON` | unset | Truthy (`1`/`true`/`yes`/`on`, case-insensitive) enables single-line JSON formatter on stderr. (`logging_config.py`) |

### Heartbeat daemon

| var | default | purpose |
|-----|---------|---------|
| `CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S` | `30.0` | Seconds `HeartbeatConnection._close` waits for the in-flight cycle before releasing the thread. Clamped to ≥ 0. (`heartbeat/connection.py`) |
| `CARL_HEARTBEAT_POLL_INTERVAL_S` | `5.0` | Sleep between queue polls when the queue is empty. Must be positive and finite; invalid → default. (`heartbeat/daemon.py`) |
| `CARL_MAINTENANCE_INTERVAL_CYCLES` | `100` | Completed-cycle trigger for in-loop maintenance. `0` disables the cycle trigger (run `carl db maintenance` out of band). (`heartbeat/loop.py`) |
| `CARL_MAINTENANCE_INTERVAL_SECONDS` | `3600.0` | Wall-clock trigger; guarantees idle daemons still truncate WAL. Fires on whichever (cycle or seconds) trips first. (`heartbeat/loop.py`) |
| `CARL_STICKY_RETENTION_DAYS` | `30` | Maintenance sweep deletes `archived` sticky notes older than N days. Negative → 0 (delete all archived). Invalid → 30. (`heartbeat/loop.py`) |

### Research / training probes

| var | default | purpose |
|-----|---------|---------|
| `CARL_CONTRACTION_PROBE` | unset | `=1` enables `ContractionProbe(window=50)` inside `ResonanceLRCallback`. Any other value is off. (`training/lr_resonance.py`) |
| `CARL_LAYER_PROBE` | unset | Truthy (`1`/`true`/`True`) flips `CARL_LAYER_PROBE_ENABLED` so callers materialize `hidden_states` + `measure_multi_layer`. Default path stays logits-only. (`carl_core/coherence_probe.py`) |

### Eval runner (HF Jobs entrypoint env)

These names are written *into* the HF Jobs entrypoint script the eval runner
generates. Set them in the job environment, not locally.

| var | purpose |
|-----|---------|
| `CARL_MODEL` | Base model ID in the generated entrypoint (`eval/runner.py`) |
| `CARL_SFT_ADAPTER` | SFT adapter ID passthrough (`eval/runner.py`) |
| `CARL_ADAPTER` | GRPO/checkpoint adapter ID passthrough (`eval/runner.py`) |
| `CARL_DATASET` | Dataset ID passthrough (`eval/runner.py`) |
| `CARL_EVAL_SAMPLES` | Integer sample cap for the eval pass (`eval/runner.py`) |

### Adapters / backends

| var | consumer | purpose |
|-----|----------|---------|
| `CARL_UNSLOTH_CONFIG` | `adapters/unsloth_adapter.py` | JSON blob consumed by the generated unsloth entrypoint script (set by the adapter itself; operators rarely set it). |
| `CARL_ADAPTER_STATE_DIR` | `adapters/_common.py` | Overrides `~/.carl/adapters/<backend>/` state root. |

### LLM provider override

| var | consumer | purpose |
|-----|----------|---------|
| `CARL_LLM_BASE_URL` | `llm.py` | Custom OpenAI-compatible endpoint. Highest priority in `LLMProvider.auto()`. |
| `CARL_LLM_MODEL` | `llm.py` | Model name override for the selected provider. |

### State root + UI

| var | default | purpose |
|-----|---------|---------|
| `CARL_HOME` | `~/.carl` | Honored by `adapters/_common.py` (`<CARL_HOME>/adapters`), `environments/registry.py` (`<CARL_HOME>/environments`), and `a2a/identity.py` (`<CARL_HOME>/keys`). NOT honored by `db.py`, `settings.py` global config, `wallet_store.py`, or `llm.py` cache — those use `Path.home()/.carl` directly. Treat `CARL_HOME` as a **partial** override limited to the call sites above. |
| `CARL_PERSONA` | unset | `carl` or `carli`; selects default theme on first run. (`theme.py`) |

### Settings-backed env vars

The Pydantic `CARLSettings` model exposes every field under the `CARL_` prefix
automatically. The most operationally relevant are:

| var | default | purpose |
|-----|---------|---------|
| `CARL_TIER` | `free` | `free` or `paid`. `pro` / `enterprise` are aliases for `paid`. (`settings.py`, `tier.py`) |
| `CARL_PRESET` | `custom` | `research` / `production` / `quick` / `custom`. Presets seed `log_level`, `default_compute`, `observe_defaults`. (`settings.py`) |
| `CARL_DEFAULT_MODEL` | `""` | Default base model for training. User must configure. |
| `CARL_DEFAULT_CHAT_MODEL` | `claude-sonnet-4-6` | `carl chat` model. Anthropic short-form IDs only. |
| `CARL_DEFAULT_COMPUTE` | `l40sx1` | Default compute target (see `ComputeTarget` enum). |
| `CARL_HUB_NAMESPACE` | `""` | HF Hub namespace. Auto-detected from `whoami` when empty. |
| `CARL_NAMING_PREFIX` | `""` | Prefix applied to generated names. User must configure. |
| `CARL_TRACKIO_URL` | unset | Trackio dashboard URL. |
| `CARL_SUPABASE_URL`, `CARL_SUPABASE_ANON_KEY` | `""` | `carl.camp` platform surface. |
| `CARL_SHOW_PER_TURN_COST` | `true` | Toggle per-turn cost delta in `carl chat`. |

## CLI operational verbs

| verb | source | purpose |
|------|--------|---------|
| `carl doctor` | `cli/startup.py::doctor` | Readiness + typed freshness report (issue codes under `carl.freshness.*`). |
| `carl start` | `cli/startup.py::start` | Guided onboarding; pairs with `carl init`. |
| `carl queue add <text> [-p N]` | `cli/queue.py::queue_add` | Append a sticky note (priority 1–10, higher runs sooner). |
| `carl queue list [-n N] [-s status]` | `cli/queue.py::queue_list` | Render notes by status. |
| `carl queue status` | `cli/queue.py::queue_status` | Bucket counts by status. |
| `carl queue clear [--done\|--all]` | `cli/queue.py::queue_clear` | Archive completed (default) or all non-archived notes. |
| `carl queue reclaim [--max-age S]` | `cli/queue.py::queue_reclaim` | Flip `processing` rows older than `--max-age` back to `queued`. Default 600s. |
| `carl db maintenance [--retention-days N] [--vacuum/--no-vacuum]` | `cli/db.py::db_maintenance` | WAL checkpoint + retention sweep. `--vacuum` takes an exclusive lock. Default retention 30 days. |
| `carl camp consent show [--json]` | `cli/consent.py::consent_show` | Display all four consent flags + change timestamps. |
| `carl camp consent update <key> [--enable/--disable]` | `cli/consent.py::consent_update` | Toggle one of `observability` / `telemetry` / `usage_analytics` / `contract_witnessing`. |
| `carl camp consent reset [-f]` | `cli/consent.py::consent_reset` | All flags off. |
| `carl-heartbeat [--db PATH]` | `heartbeat/daemon.py::main` | Standalone 24/7 worker; `pyproject.toml` registers it under `[project.scripts]`. |

## Daemon runbook

- **Deployment:** systemd unit, Docker `ENTRYPOINT`, or container-orchestrator `command` invoking `carl-heartbeat`. Console script is registered in `pyproject.toml`.
- **Boot reclaim:** on startup, `daemon.py::_run` calls `StickyQueue.reclaim_stale()` *before* the loop dequeues. Rows left in `processing` by a SIGKILL are flipped back to `queued`. Without this the SIGKILL survivors would sit wedged forever while the loop drained fresh `queued` work past them (ticket E2).
- **SIGTERM / SIGINT:** POSIX paths use `loop.add_signal_handler`; Windows falls back to `signal.signal` + `loop.call_soon_threadsafe`. Both flip the same `asyncio.Event`, so shutdown is uniform. The in-flight cycle runs to completion (no mid-phase `stop_event` check). `HeartbeatConnection._close` joins with `CARL_HEARTBEAT_SHUTDOWN_TIMEOUT_S` budget.
- **Maintenance cadence:** fires on whichever trigger trips first — `CARL_MAINTENANCE_INTERVAL_CYCLES` (cycle-count) OR `CARL_MAINTENANCE_INTERVAL_SECONDS` (wall-clock). Time-based trigger guarantees idle daemons still checkpoint WAL + run retention.
- **Cycle failure:** exception mid-phase → `StickyQueue.requeue(note.id)` flips the row back to `queued` for retry. Happy path → `StickyQueue.complete(id, result)` transitions to `done`. A cycle never leaves a note wedged in `processing`.
- **Exit codes:** `0` on graceful shutdown (signal received). `1` on startup failure (DB unusable, signal handler install failed).

## Consent enforcement

The four flags on `ConsentManager` are runtime-enforced gates — call sites
invoke `consent_gate(flag)` at the network boundary. All default to OFF.

| flag | gates |
|------|-------|
| `telemetry` | `carl_studio.sync.push` / `.pull`, MCP `authenticate` tool |
| `contract_witnessing` | `X402Client.execute`, `PaymentConnection.get` (x402 payments create service-contract witnesses) |
| `observability` | reserved — outbound coherence-probe publish paths |
| `usage_analytics` | reserved — future analytics emission |

Grant with `carl camp consent update <flag> --enable`. Revoke with `--disable`
or `carl camp consent reset`.

## Supply chain

- **Lockfile:** `uv.lock` at repo root. CI enforces `uv lock --check` and `uv sync --locked --all-extras` before the build step. (`.github/workflows/publish.yml`)
- **SBOM:** cyclonedx-bom generates `sbom.json` on every PyPI release (`cyclonedx-py environment`). Uploaded as a release artifact and attached to the GitHub Release via `gh release upload`.
- **Extras matrix:** see `docs/INSTALL.md`. `[quickstart]` is the recommended default for new users; `[all]` pulls every optional dependency.
- **Release workflow:** triggered by GitHub Release (root) or scoped tag push `carl-<pkg>@<version>` (sub-packages). Trusted Publisher via `pypa/gh-action-pypi-publish`.

## Metrics

v0.6.x ships no `/metrics` endpoint; Prometheus scrape is deferred. Until
then, the canonical operational signal is structured JSON logs:

```bash
CARL_LOG_JSON=1 CARL_LOG_LEVEL=info carl-heartbeat
```

Every `HeartbeatLoop._run_maintenance` call emits a `heartbeat.maintenance`
status record (`reclaimed`, `notes_deleted`, `wal_checkpoint`). Every cycle
emits `cycle:start` / `cycle:end` steps on the shared `InteractionChain` with
action type `HEARTBEAT_CYCLE`.
