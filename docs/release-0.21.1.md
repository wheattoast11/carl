---
last_updated: 2026-10-08
author: CARL project
applies_to: carl-studio 0.21.1
---

# CARL 0.21.1 verification

Source base: `e10b974f3e2ff05ad1daadb474802f117f5adf27` on
`codex/carl-prepared-training`. The existing HF Jobs edit was retained and
extended with token redaction and complete env/secret collision removal.
Public main was `5be38462a9c3dba089e24f1386a945dc2469b99c` at release preparation.

## Changes

- Private HF Jobs script URLs use URL-encoded token authentication on raw paths.
  CARL logs repository and script identity without the token. Submission errors
  redact raw and encoded token values and suppress the sensitive SDK traceback.
- Environment entries take precedence over secret mappings for every key.
- Legacy project `backend` loads as `adapter`; an explicit `adapter` wins.
  Project inspection keeps compute orchestration distinct from the adapter.
- Relative preparation datasets resolve beside the config before the discovered
  project root. The config is bound so prepared-plan validation uses the same rule.
- Source hashing uses at most four workers. Checkpoint digests are cached by
  device, inode, size, mtime and ctime. The first binding reads all bytes;
  execution validation continues to rehash every source. An index-only digest
  was excluded because it would miss changes to shard contents.
- Training checks per-sample `private_keys` and optional `trajectory` metadata,
  plus each completion. Leaks zero task and coherence contributions and receive
  an unweighted `-3.0` terminal reward outside cascade masks. Evaluation can
  enforce the same event boundary with `EvalConfig.private_keys`.
- Plugin updates remove retired owned references, including upgrades from the
  previous ownership format. Edited files are refused and preserved.
- Canonical skill workflows document preparation and privacy metadata.

## Executed checks

- `.venv/bin/python -m pytest tests/ packages/carl-core/tests/ -q --tb=short`:
  **4,411 passed, 29 skipped, 85 warnings**, 93.14 seconds.
- Strict Pyright for plugin, preparation, privacy rewards and project loading:
  **0 errors**.
- Ruff for those modules and new plugin/privacy/project regressions: **passed**.
- Comparative checks for the other touched modules: Ruff **74 baseline and
  74 current diagnostics**; Pyright **248 baseline and 230 current diagnostics**;
  **no introduced diagnostics**. Baselines were extracted from the source base.
- `uv lock --check --offline`: **passed**, 372 packages, no dependency upgrades.
- `python -m build --outdir dist/v0.21.1`: source archive and wheel **built**.
- Fresh isolated environment installed the wheel and existing core 0.3.0 wheel
  offline. CLI version, module locations, legacy adapter loading, terminal privacy
  penalties, malformed events, numeric identifiers, and lightweight imports
  **passed**. Torch and Transformers were absent from the import path exercised.
- Installed Codex, Claude Code and OpenCode completed native metric witnesses
  against synthetic localhost providers. All returned matching artifacts and
  bound native sessions. All three cancellation witnesses stopped execution
  without cancellation errors.
- Source archive contains canonical plugin manifests and skill references;
  wheel contains the privacy reward module. Artifact hashes: `dist/v0.21.1/SHA256SUMS`.

## Scope

HF Jobs regressions use mocked SDK calls. No paid HF job or model training was
submitted. Native witnesses use synthetic localhost providers. The 55 GB
checkpoint from the integration report was not benchmarked. Cold checkpoint
binding still requires reading its bytes; repeated preparation uses the cache.
GitHub and PyPI publication are separate from these local checks.
