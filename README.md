<p align="center">
  <img src="assets/carl-hero.gif" alt="CARL: from chaos to crystal" width="720"/>
</p>

<h1 align="center">CARL</h1>

<p align="center">
  <em>Coherence-Aware Reinforcement Learning</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/carl-studio/"><img src="https://img.shields.io/pypi/v/carl-studio?color=06b6d4&style=flat-square" alt="PyPI"/></a>
  <a href="https://pypi.org/project/carl-studio/"><img src="https://img.shields.io/pypi/pyversions/carl-studio?style=flat-square" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/></a>
  <a href="https://doi.org/10.5281/zenodo.18906944"><img src="https://img.shields.io/badge/paper-Zenodo-blue?style=flat-square" alt="Paper"/></a>
</p>

---

## Why

A model becomes an agent when it stops pattern-matching and starts *knowing*. That transition isn't gradual — it's a **phase transition**, like water becoming ice. One moment the model is guessing. The next, it's coherent.

Standard training can't see this happening. You watch a loss curve and hope.

**CARL measures the moment of crystallization** — and rewards it.

```
                         Phi (order parameter)
                              │
          guessing            │         knowing
     ░░░░░░░░░░░░░░░░░░░░░░░░│████████████████████████
                              │
                        crystallization
```

The order parameter **Phi** measures how coherent a model's probability field is at every token. When Phi crystallizes, the model has found its internal anchor — a fixed point it can navigate from to *any* concept space without losing itself.

This is alignment you can measure, not just evaluate.

---

## Install

```bash
pip install carl-studio                     # core CLI + one-shot observe
pip install 'carl-studio[training]'        # local train/eval
pip install 'carl-studio[hf]'              # status/logs/stop/push/HF Jobs
pip install 'carl-studio[tui]'             # observe --live
pip install 'carl-studio[observe]'         # observe --diagnose
pip install 'carl-studio[wallet]'          # x402 / Coinbase wallet
pip install 'carl-studio[all]'             # everything
```

Most users should start with:

```bash
pip install 'carl-studio[training,hf]'
```

## Quickstart

```bash
pip install carl-studio
carl init                  # one-shot setup: account, provider, extras, project, consent
carl "train a small model on gsm8k"   # agent — one-shot prompt
carl chat                  # agent — interactive loop
```

`carl init` is idempotent: re-running it after setup does nothing unless you pass `--force`. A first-run marker lives at `~/.carl/.initialized`.

## Auth

CARL Studio does not require a `.env`, and it does not auto-load one.

- Hugging Face workflows work with either `HF_TOKEN` or a prior `hf auth login` / `huggingface-cli login`
- Claude-powered features use `ANTHROPIC_API_KEY` or `--api-key`
- RunPod uses `RUNPOD_API_KEY`
- public Trackio observe works without credentials

If you want a template, copy `.env.example` and load it into your shell before running `carl`:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Quick setup:

```bash
hf auth login
export ANTHROPIC_API_KEY=sk-ant-xxx   # only for --diagnose / chat
carl start
```

Full auth details: [`docs/auth.md`](docs/auth.md)

## Primary commands

| Command | What it does |
|---|---|
| `carl init` | One-shot setup: account, provider, extras, project, consent. |
| `carl chat` | Interactive agent loop with tools, sessions, cost tracking. |
| `carl "<prompt>"` / `carl ask "<prompt>"` | One-shot agent invocation. |
| `carl flow "/a /b /c"` | Chain named operations, emit a shared interaction trace. |
| `carl doctor` | Readiness audit. Prints blocking issues and freshness findings. |
| `carl train` | Local training with coherence rewards (`carl-studio[training]`). |

Run `carl start --inventory` for the full installed command map, or `carl flow --list` for every chainable op.

## Architecture

- **`carl-core`** — primitive layer. Typed errors, retry/backoff, safepath sandboxing, content hashing, tier gating, coherence math, interaction chains. Zero training deps.
- **`carl-studio`** — the CLI, agent loop, training pipeline, MCP server, camp client, eval sandbox. Everything above builds on `carl-core`.

`carl-core` is installed alongside `carl-studio`; public callers import from `carl_core.*` directly. The legacy `carl_studio.primitives` shim was removed after v0.5.0.

## Error contract

Fatal paths raise `carl_core.errors.CARLError` subclasses with stable codes you can match programmatically. Top codes:

| Code | Meaning |
|---|---|
| `carl.error` | Base class. Generic failure. |
| `carl.config` | Invalid or missing configuration. |
| `carl.validation` | Input failed schema / value validation. |
| `carl.credential` | Missing or expired credential. |
| `carl.network` | Transient or persistent network failure. |
| `carl.budget` | Spend cap exceeded. |
| `carl.permission` | Permission / consent gate failed. |
| `carl.timeout` | Operation exceeded its deadline. |
| `carl.freshness.stale_pkg` | Installed package older than recommended floor. |
| `carl.freshness.camp_session_expired` | `carl.camp` session needs `carl camp login`. |

`CARLError.to_dict()` produces a secrets-redacted, telemetry-safe payload. See `packages/carl-core/src/carl_core/errors.py` for the full hierarchy.

## Use

**See inside a Trackio run** (no GPU required, base install):
```bash
carl observe --url https://your-trackio-space.hf.space/ --run your-run
```

If the dashboard contains multiple projects, add `--project your-project`.

**Train with coherence rewards** (`carl-studio[training]`):
```bash
carl project init
carl train --config carl.yaml
carl run list
```

Or run directly from the CLI:

```bash
carl train --model your-org/your-base-model --method grpo --dataset your-org/your-dataset --output-repo your-org/your-model --compute a100-large
```

**Gate a checkpoint** (`carl-studio[training]`):
```bash
carl eval --adapter your-username/your-model
```

---

## How It Works

```
 ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌──────┐     ┌──────┐
 │ Observe │ ──> │ Measure │ ──> │  Train  │ ──> │ Gate │ ──> │ Ship │
 │         │     │   Phi   │     │  CARL   │     │      │     │      │
 └─────────┘     └─────────┘     └─────────┘     └──────┘     └──────┘
  point at        entropy +       task rewards     cascade      push to
  any run         order param     + coherence      auto-fires   hub
```

**Observe** — Point CARL at a Trackio dashboard or log file. Instantly see Phi trajectory, entropy, phase state, health.

**Measure** — Phi = 1 - H(P)/log|V|. Zero means maximum uncertainty. One means complete coherence. Computed per token, every step.

**Train** — Five reward functions in a cascade. Task rewards teach *what*. CARL rewards teach *how coherently*.

**Gate** — The cascade auto-calibrates from the training signal. No hardcoded thresholds. CARL activates only when the model demonstrates sustained capability.

**Ship** — Eval gate passes → checkpoint pushed to Hub.

---

## CLI Install Matrix

| Workflow | Command | Install |
|---|---|---|
| One-shot observe | `carl observe --url ... --run ...` | `pip install carl-studio` |
| Live observe | `carl observe --live ...` | `pip install 'carl-studio[tui]'` |
| Claude diagnosis | `carl observe --diagnose ...` | `pip install 'carl-studio[observe]'` |
| Local train/eval | `carl train`, `carl eval` | `pip install 'carl-studio[training]'` |
| HF job management / publish | `carl run status`, `carl run logs`, `carl run stop`, `carl push` | `pip install 'carl-studio[hf]'` |
| Camp account + marketplace | `carl camp account`, `carl camp login`, `carl camp logout`, `carl camp credits`, `carl camp marketplace` | platform features (optional) |
| Privacy consent | `carl camp consent show`, `carl camp consent update` | included |
| x402 payment rail | `carl camp x402 configure`, `carl camp x402 status` | included |
| Contract witnessing | `carl camp contract sign`, `carl camp contract verify` | included |
| Carlito management | `carl carlito list`, `carl carlito spawn`, `carl carlito show` | included |

Managed tiers build on top of these open workflows; extras control local capabilities, not research access.

Provider credentials unlock provider workflows, not CARL Paid platform access. Use `carl camp account` to inspect managed account state, credits, and enabled wallet/x402 capabilities. Privacy consent is managed locally with `carl camp consent` — all flags default off.

## Credential Matrix

| Workflow | Auth |
|---|---|
| Local file observe | none |
| Public Trackio observe | none |
| Claude diagnosis / chat | `ANTHROPIC_API_KEY` or `--api-key` |
| Hub jobs / push / gated model access | `HF_TOKEN` or prior HF login |
| RunPod backend | `RUNPOD_API_KEY` |

---

## Results

Trained with CARL on [OmniCoder-9B](https://huggingface.co/Tesslate/OmniCoder-9B):

| Metric | Value |
|--------|-------|
| Task completion | **92%** |
| Tool format compliance | 99% |
| Mean tool calls per task | 11.09 |
| Phase 2' eval gate | **PASS** |

80 GRPO steps. Five reward functions. Self-calibrating cascade gate.

---

## Papers

The math is published and independently reproducible:

- [Bounded Informational Time Crystals](https://doi.org/10.5281/zenodo.18906944) — derives the conservation law
- [Material Reality](https://doi.org/10.5281/zenodo.18992029) — validates across 6,244 trials
- [Semantic Realizability](https://doi.org/10.5281/zenodo.18992031) — formal proof

---

## Reference

Architecture, API, CLI commands, environments, compute backends → [docs/reference.md](docs/reference.md)

Credential setup and provider auth → [docs/auth.md](docs/auth.md)

---

## Star History

<a href="https://star-history.com/#wheattoast11/carl&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=wheattoast11/carl&type=Date&theme=dark"/>
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=wheattoast11/carl&type=Date"/>
    <img alt="Star History" src="https://api.star-history.com/svg?repos=wheattoast11/carl&type=Date"/>
  </picture>
</a>

---

<p align="center">
  <a href="https://terminals.tech">terminals.tech</a> · <a href="https://pypi.org/project/carl-studio/">PyPI</a> · <a href="https://doi.org/10.5281/zenodo.18906944">Paper</a> · <a href="docs/reference.md">Docs</a>
  <br/><br/>
  MIT — <a href="https://terminals.tech">Intuition Labs LLC</a>
</p>
