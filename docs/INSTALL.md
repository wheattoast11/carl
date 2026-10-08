---
last_updated: 2026-10-07
author: Tej Desai
applies_to: carl-studio 0.20.1
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
| `mcp` | MCP server (`carl-mcp` / `carl lab mcp`) | `pip install 'carl-studio[mcp]'` |
| `research` | `carl research ...` and `carl lab research ...` (arxiv) | `pip install 'carl-studio[research]'` |
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
- `x402` — requires `x402>=2.25` (the newer standalone rail)

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

## October 7 compatibility snapshot

Python 3.12+ is the package baseline. One `uv.lock` serves the lightweight and
all-extra installs. The installer preserves existing extras in the same project
environment. SGLang is included on Linux Python 3.12/3.13; other platforms keep
the adapter's documented dependency-availability behavior.

SGLang 0.5.21 constrains Torch to 2.13.0, Transformers to 5.12.1, and tokenizers
to 0.22.2. Hub stays at 1.33.0. Mistral requires NumPy below 2.4 on Python 3.12,
so that interpreter uses 2.3.5; Python 3.13 uses 2.4.6 and Python 3.14+ uses 2.5.3.
CUDA Tile 1.6.0rc5 and FlashAttention 4 beta are required upstream exceptions.

Legacy wallet support remains opt-in on Python 3.12/3.13. Its AgentKit/NilQL/BCL
graph retains older CFFI, cryptography, PyNaCl, Paramiko, and x402 versions. It
cannot be combined with all, x402, RunPod, constitutional, or secrets extras.
Python 3.14+ retains wallet encryption/state support without AgentKit creation.
Exact advisory matches and applicability are recorded in the dependency review.

## Native plugin witnesses

```bash
python scripts/witness_native_harness.py codex metrics
python scripts/witness_native_harness.py claude metrics
python scripts/witness_native_harness.py opencode metrics
```

These use actual installed binaries, synthetic credentials, disposable profiles,
and localhost providers. `cancel` replaces `metrics` to exercise interruption.
They require local socket access and do not qualify a paid provider or GPU model.
