---
last_updated: 2026-10-07
author: CARL project
applies_to: carl-studio 0.20.1
---

# Dependency review

Release snapshot: October 7, 2026, America/Chicago (`2026-10-08T00:35:32Z`).
Exact registry versions, release metadata, advisory IDs, and the lock digest are
in `dependency-audit-2026-10-07.json`. The audit covers 371 registry entries.
An advisory lookup is evidence about published matches, not a security guarantee.

## Selected compatibility graph

One `uv.lock` covers the base, developer, and all-extra environments.
The complete Linux Python 3.12 all-extra graph installed successfully, including
CUDA Tile's required source build. Torch CPU tensors, TorchAudio resampling,
SFTConfig/GRPOConfig construction, and SGLang import executed successfully.
No GPU model execution or training was part of those checks.

SGLang 0.5.21 requires Torch 2.13.0, Transformers 5.12.1, tokenizers 0.22.2,
CUDA Tile 1.6.0rc5, and FlashAttention 4 beta. Transformers holds Hub below 2;
the selected Hub version is 1.33.0. Mistral holds NumPy below 2.4 on Python 3.12,
which uses 2.3.5; Python 3.13 uses 2.4.6 and Python 3.14+ uses 2.5.3.
OpenTelemetry's upstream semantic-convention packages also use beta versions.

MCP 2.3.0 uses public MCPServer APIs and protocol-era resolvers. Anthropic 1.12.0
uses its exported HTTP timeout type. TypeScript 7 requires explicit source roots,
Node types, and Node16 resolution for CommonJS output. Golden codec/signature
vectors remain unchanged.

## Retained advisory matches

| Dependency | Selection | Applicability and disposition |
| --- | --- | --- |
| cryptography 45.0.7 | Legacy wallet only | AgentKit/NilQL/BCL holds CFFI below 2. CARL wallet storage uses Fernet and PBKDF2; PKCS#7 recipient decryption is not its path. Other matched operations remain listed in the audit. Default graph uses 50.0.2. |
| PyNaCl 1.6.0 | Legacy wallet only | Same CFFI constraint. The advisory concerns Ed25519 point validation. Default constitutional/secrets graph uses fixed 1.6.2. |
| Paramiko 3.5.1 | Legacy wallet only | AgentKit holds Paramiko below 4. Its advisory concerns SHA-1 SSH signatures. Default RunPod graph uses 5.0.0. |
| x402 1.0.0 | Legacy wallet only | AgentKit holds x402 below 2. The advisory concerns Solana facilitators; CARL's inspected standalone SDK path registers an EVM signer. Default graph uses 2.25.0. |
| ecdsa 0.19.2 | Wallet/payment transitive graph | The timing advisory concerns P-256 signing. No patched release was available in the snapshot. A match does not establish CARL execution of that signing path. |
| diskcache 5.6.3 | SGLang/tooling transitive graph | No patched release was available. Pickle deserialization is unsafe when another actor can write cache contents. Keep backend caches operator-owned and do not consume untrusted cache directories. |

Legacy wallet remains opt-in and incompatible with all, modern x402, RunPod,
constitutional, and secrets extras. Its retained constraints are explicit rather
than resolver overrides. Existing encrypted wallet state and public APIs remain.

## Reproduction

```bash
python scripts/audit_dependencies.py --output build/dependency-audit.json
uv lock --check
uv sync --locked --extra all
```

The audit records its actual query timestamp separately from the fixed release
snapshot. Recheck the resolved graph before a release; later advisory publication
can change the result without changing package versions.
