---
last_updated: 2026-04-19
author: Tej Desai
applies_to: v0.8.0
---

# Install

`carl-studio` ships a minimal core + a matrix of optional extras. Pick what you need.

## Default (most users)

```bash
pip install 'carl-studio[quickstart]'
```

Bundles `training` + `hf` + `observe`. That is the three-way combo ~95% of users
actually run with. Under the hood:

- `training` — local train/eval loop (torch, transformers, trl, peft, datasets, bitsandbytes, trackio)
- `hf` — Hugging Face Hub job management and publish
- `observe` — Claude-powered diagnosis (`carl observe --diagnose`)

## Bare install

```bash
pip install carl-studio
```

Ships the CLI, core types, `carl-core`, and one-shot Trackio observe. No GPU
dependencies, no LLM clients. Safe for CI or a laptop that only needs to poke
at a remote run.

## Full extras matrix

| Extra | What it enables | Install |
|---|---|---|
| `training` | Local training + eval + GRPO rewards | `pip install 'carl-studio[training]'` |
| `hf` | Hugging Face Hub job status / logs / stop / push | `pip install 'carl-studio[hf]'` |
| `observe` | Claude `--diagnose` flag | `pip install 'carl-studio[observe]'` |
| `tui` | `carl observe --live` (Textual TUI) | `pip install 'carl-studio[tui]'` |
| `runpod` | RunPod compute backend | `pip install 'carl-studio[runpod]'` |
| `tinker` | Tinker compute backend | `pip install 'carl-studio[tinker]'` |
| `mcp` | MCP server (`carl mcp serve`) | `pip install 'carl-studio[mcp]'` |
| `research` | `carl research` (arxiv) | `pip install 'carl-studio[research]'` |
| `a2a` | Agent-to-agent protocol | `pip install 'carl-studio[a2a]'` |
| `wallet` | Coinbase AgentKit wallet + keyring | `pip install 'carl-studio[wallet]'` |
| `x402` | x402 HTTP payment rail (standalone, newer) | `pip install 'carl-studio[x402]'` |
| `payments` | Stripe Agent Toolkit | `pip install 'carl-studio[payments]'` |
| `dev` | pytest / ruff / pyright / hypothesis | `pip install 'carl-studio[dev]'` |
| `all` | Everything except `wallet` (see Conflicts below) | `pip install 'carl-studio[all]'` |

## Conflicts

Two extras are **mutually exclusive** because their upstream dependency graphs
disagree on a pinned version of `x402`:

- `wallet` — pulls `coinbase-agentkit`, which pins `x402<2`
- `x402` — requires `x402>=2.7` (the newer standalone rail)

You must pick one. The `[all]` meta-extra includes `x402` (the newer rail) and
deliberately excludes `wallet`. If you need Coinbase AgentKit, install it
separately without `[all]`:

```bash
pip install 'carl-studio[training,hf,wallet]'
```

The conflict is declared in `[tool.uv]` so `uv lock` can resolve, and the
freshness check in `carl doctor` reports if both are detected in a single
environment.

## Reproducible installs

The repo ships a committed `uv.lock` pinning the full dependency graph for the
`[all]` configuration. For byte-reproducible installs:

```bash
uv sync --locked --extra all
```

CI enforces the lockfile with `uv lock --check` before publish. An SBOM
(CycloneDX) is emitted on every release and attached to the GitHub release as
`sbom.json`.

## Credentials

None of the install paths require credentials. Runtime credentials (HF, Claude,
RunPod, Stripe, etc.) are described in [`docs/auth.md`](auth.md).
