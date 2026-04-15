# CARL Studio Auth

CARL Studio does not require a `.env` file, and it does not auto-load one.

You can authenticate in either of these ways:

- export environment variables in your shell
- log into Hugging Face locally with `hf auth login` / `huggingface-cli login`

If you want to keep keys in a `.env`, copy `.env.example` and load it yourself before running `carl`:

```bash
cp .env.example .env
set -a
source .env
set +a
```

## Credential Matrix

| Workflow | What you need |
|---|---|
| `carl observe --file ...` | nothing |
| `carl observe --url ... --run ...` against a public Trackio Space | nothing |
| `carl observe --live ...` | no key required for public Spaces; install `carl-studio[tui]` |
| `carl observe --diagnose ...` | `ANTHROPIC_API_KEY` or `--api-key` |
| `carl lab chat` | `ANTHROPIC_API_KEY` or `--api-key` |
| `carl train`, `carl eval` on Hub models/datasets | `HF_TOKEN` or prior Hugging Face login |
| `carl run status`, `carl run logs`, `carl run stop`, `carl push` | `HF_TOKEN` or prior Hugging Face login |
| RunPod backend | `RUNPOD_API_KEY` and usually `HF_TOKEN` |

## Provider Notes

### Hugging Face

CARL Studio uses the `huggingface_hub` SDK for Hub access, job management, pushing artifacts, and loading private/gated repos.

Supported auth paths:

```bash
hf auth login
```

or:

```bash
export HF_TOKEN=hf_xxx
```

In practice, either is fine. Different code paths consult `HF_TOKEN` and cached Hub credentials, so logging in once with the HF CLI is enough for many users.

### Anthropic

Anthropic is only used for Claude-powered features.

```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
```

Or pass a key directly:

```bash
carl observe --file logs/train.jsonl --diagnose --api-key sk-ant-xxx
carl lab chat --api-key sk-ant-xxx
```

### RunPod

The RunPod backend reads:

```bash
export RUNPOD_API_KEY=rp_xxx
```

If the training job also needs Hugging Face model or dataset access, set `HF_TOKEN` too.

### Trackio

Remote observe uses Gradio client calls against a Trackio Space.

- public Trackio Spaces work without credentials
- local file observe works without credentials
- the CLI currently assumes public/shareable Trackio Spaces for remote observe

## Config Files vs Secrets

`carl config init` and `~/.carl/config.yaml` are for non-secret defaults like:

- default model
- default compute target
- Hub namespace
- Trackio URL

Secrets are intentionally not persisted there.

Use this to verify what CARL Studio detected:

```bash
carl config show
```

To see raw values instead of masked output:

```bash
carl config show --unmask
```

## Recommended Setup

For most users:

```bash
pip install 'carl-studio[training,hf]'
hf auth login
carl config init
carl config show
```

If you also want Claude-powered diagnosis/chat:

```bash
pip install 'carl-studio[training,hf,observe]'
export ANTHROPIC_API_KEY=sk-ant-xxx
```
