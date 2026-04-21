---
last_updated: 2026-04-21
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.16+ (design pending user sign-off before implementation)
---

# v0.16+ secrets toolkit + computer-use — design

**Status:** design synthesis from three parallel research agents
dispatched 2026-04-21. Awaiting user decision on scope + staging
before any code lands.

## The problem Tej described

> a virtual KVM seems like the missing link ... as long as we have an
> agent that can make decisions and use that tooling/harness effectively
> and has no way to do the same task without using that tool, thereby
> constraining and gating the risk out entirely and not making it a
> factor in the system of operations

Concretely: Carl should be able to complete workflows like
"generate an API key → paste it into this form → submit" **without
ever seeing the value**. The value flows through OS-level primitives
(encrypted store → clipboard → GUI paste) that Carl can invoke via
opaque handles but cannot dereference into its own context.

The agent's epistemic state of the value stays ⊥ throughout the
entire workflow. Only a 12-hex fingerprint lands in the
`InteractionChain` audit trail. Risk is structurally constrained to
the tool primitives, not policed at the prompt level.

## Research findings (three agents, 2026-04-21)

### 1. Anthropic computer-use primitives (MIT)
- 28 actions across `computer_20241022` / `20250124` / `20251124`.
- **No zero-knowledge primitives.** Clipboard not exposed by design.
- Reference `loop.py` + coordinate-scaling are directly liftable.
- Everything we need below Anthropic's surface (secret handles,
  scoped clipboard) is work we build ourselves.

### 2. OSS computer-use landscape (audit April 2026)
- **browser-use** (MIT, 89k★) — adopt as optional extra for browser.
- **playwright-python** (Apache-2.0, 14k★) — adopt as dep.
- **OpenAdapt** (MIT, 1.6k★) — trajectory shape matches `InteractionChain`.
  Sub-package `openadapt-privacy` scrubs PII on screenshots — fork.
- **Cua / trycua** (MIT, 13k★) — adopt as sandboxed VM backend.
- **Skyvern** (AGPL) — STUDY the credential-integration pattern;
  reimplement under MIT from the contract, don't vendor the code.
- **No OSS project** in the survey implements capability-security /
  handle-based secret transit. That is CARL's differentiator.

### 3. Zero-knowledge primitives (capability-security patterns)
- **carl-studio already has** `wallet_store.py` (Fernet + PBKDF2-HMAC-
  SHA256 at 600k iters + `keyring` + mode 0600). Don't reinvent the
  crypto — build the handle layer above it.
- **Recommended primitive stack:**
  - `pynacl` sealed boxes (already in `[constitutional]`) — for
    agent-produced secrets the agent can't read back.
  - `cryptography.fernet` (already in `[wallet]`) — user-keyed envelopes.
  - `keyring` (already transitive) — OS keychain bridge.
  - `pyperclip` — clipboard wrap.
  - Thin `ctypes`/`subprocess` wrappers for scoped-paste (no library).
- **Prior art:** HashiCorp Vault "response wrapping" + AWS STS
  AssumeRole (handle-based). 1Password `op://` URI shape. No Python
  OCAP framework is production-ready — idiomatic is enough (60 LOC).

## Proposed architecture

### Layer 1: `carl_core.secrets` (MIT, carl-core)

Pure primitives. No network, no platform-specific code. Lives in
carl-core so downstream packages import without circular deps.

```python
# SecretRef — opaque handle. No value attribute.
class SecretRef(BaseModel):
    ref_id: UUID
    kind: Literal["env","keychain","vault","clipboard","mint"]
    uri: str                # "op://vault/key" shape
    ttl_s: int | None
    created_at: datetime

# SecretVault — process-local in-memory KV.
class SecretVault:
    def put(self, value: bytes, *, kind: str, ttl_s: int | None) -> SecretRef
    def resolve(self, ref: SecretRef) -> SecretBytes  # PRIVILEGED only
    def revoke(self, ref: SecretRef) -> bool
    def fingerprint(self, ref: SecretRef) -> str  # 12-hex sha256

# seal/unseal — pynacl.SealedBox wrapper.
def seal(pubkey: bytes, value: bytes) -> bytes
def unseal(privkey: bytes, ciphertext: bytes) -> bytes

# fingerprint — formalizes InteractionChain.probe_call convention.
def fingerprint(data: str | bytes) -> str  # 12-hex
```

### Layer 2: `carl_studio.secrets` (MIT, carl-studio)

Platform integrations. Wraps existing `wallet_store` + `keyring` +
`pyperclip`; extends `InteractionChain` with new audit events.

```python
# KeychainBackend — keyring wrapper with name-only semantics.
class KeychainBackend:
    def store(self, name: str, value: bytes) -> SecretRef  # name, not value
    def load(self, name: str) -> SecretRef
    def delete(self, name: str) -> bool

# ClipboardBridge — pyperclip + TTL auto-wipe + audit.
class ClipboardBridge:
    def write_from_ref(self, ref: SecretRef, *, ttl_s: int = 30) -> None
    def wipe(self) -> None  # explicit clear
    def was_modified_since(self, t0: datetime) -> bool  # integrity check

# CryptoRandomMinter — generate secrets directly into the vault.
class CryptoRandomMinter:
    def mint_hex(self, nbytes: int) -> SecretRef
    def mint_base64(self, nbytes: int) -> SecretRef
    def mint_uuid(self) -> SecretRef
    def mint_ed25519_keypair(self) -> tuple[SecretRef, SecretRef]  # (priv, pub)
```

Existing `wallet_store.WalletStore` is composed, not duplicated.

### Layer 3: Tool-dispatcher integration (`tool_dispatcher.py`)

Carl's existing `ToolDispatcher.execute_block` gains five new
tools. All take `SecretRef` by id, never raw values:

- `mint_secret(kind, length)` → `SecretRef`
- `copy_to_clipboard(ref, ttl_s)` → `{fingerprint, expires_at}`
- `revoke_secret(ref)` → `bool`
- `hash_value(ref, algorithm)` → `{fingerprint, algorithm}` — derives a
  hash through the vault without returning the raw value
- `list_secrets()` → `list[{ref_id, kind, fingerprint, ttl_remaining_s}]`
  (no URIs, no values)

New `ActionType` enum values (added to `carl_core.interaction.ActionType`):
- `SECRET_MINT`
- `SECRET_RESOLVE`
- `SECRET_REVOKE`
- `CLIPBOARD_WRITE`

Every privileged `resolve()` emits a `SECRET_RESOLVE` Step with the
ref_id, fingerprint, and the specific tool that requested the resolve
— so the audit trail shows who asked and when, but not the value.

### Layer 4: Computer-use tools (`carl_studio.cu`)

Thin wrappers over `playwright` + Anthropic computer-use schema:

- `browser.click(selector)` / `browser.type(text)` (normal text, no refs).
- `browser.type_from_ref(ref, field_selector)` — the zero-knowledge
  primitive. Opens a scoped Playwright context, derefs the ref INSIDE
  the browser process, types via DOM `input.value = ...`, immediately
  revokes the ref. Python context never holds the value past one call.
- `browser.paste_from_clipboard(field_selector)` — triggers `Cmd+V` /
  `Ctrl+V` via Playwright keyboard API; the browser reads from OS
  clipboard, Carl's Python process never touches the bytes.
- `browser.screenshot(scrub_pii: bool = True)` — forks
  `openadapt-privacy` for on-the-fly PII scrubbing before the
  screenshot hits the `InteractionChain`.

Anthropic computer-use tool schema compatibility: `cu_tool()` adapter
that exposes the same 28-action surface so Carl agents trained on
Anthropic's examples work out of the box.

## Staging plan

| Stage | Scope | LOC est | Depends on |
|---|---|---|---|
| A | `carl_core.secrets` primitives + `ActionType` extensions | ~400 | nothing |
| B | `carl_studio.secrets` + tool-dispatcher integration | ~600 | Stage A |
| C | `carl_studio.cu` browser primitives + scrub fork | ~800 | Stage B |
| D | Docs + Carl's system prompt + examples | ~300 | A + B + C |

**Optional extras** (gated on user interest, added per-need):
- `[secrets]` extra: `pynacl>=1.5`, `keyring>=24`, `pyperclip>=1.9`,
  `cryptography>=42`. All already in other extras; this is an
  aggregation for the Dispatcher-Tools-ready install.
- `[cu]` extra: `playwright>=1.58`, `browser-use>=X`. Heavy (installs
  Chromium); opt-in only.

## Decision points (need user input before code)

1. **Stage sequencing.** Ship A + B as one unit, or land A alone, roll
   out to the community for feedback, then ship B + C?
   Recommendation: A + B together (same commit), C separately.
   Rationale: A is useless without B; A + B together gives a
   complete `mint → clipboard → revoke` cycle that's independently
   valuable even without computer-use.

2. **Tier gating.** New feature keys proposed:
   - `secrets.mint` → FREE (capability)
   - `secrets.resolve` → FREE (capability)
   - `secrets.clipboard` → FREE (capability)
   - `secrets.audit` → FREE (capability)
   - `cu.browser` → FREE (capability, user runs locally)
   - `cu.managed_sandbox` → PAID (carl.camp-hosted Playwright/Cua VMs)
   Confirm this split matches the tier.py:16-24 philosophy.

3. **Audit trail verbosity.** Every `resolve()` emits a Step — that's
   potentially noisy for a large script. Option: batch same-ref
   resolves within a 1-second window into a single Step with a
   `resolve_count`. Default: per-resolve Step for cleanest audit.

4. **Clipboard TTL default.** 30s proposed. Short enough that a
   leaked clipboard is a narrow window; long enough for human-speed
   paste. Alternatives: 10s (tight), 60s (lenient).

5. **`openadapt-privacy` fork location.** Separate repo
   (`terminals-tech/carl-privacy-scrub`) or in-tree module
   (`carl_studio.privacy`)? In-tree is simpler; separate repo is
   cleaner for PR upstream if OpenAdapt accepts patches.
   Recommendation: in-tree for v0.16 ship, extract later if we
   accumulate CARL-specific changes.

6. **Computer-use Anthropic compatibility.** Ship day-1 with the
   Anthropic tool schema so users can drop in existing computer-use
   prompts, or wait and ship our own schema first?
   Recommendation: ship compat from day-1 (lifts free from
   `computer-use-demo/tools/computer.py`, MIT).

## Cross-system invariants preserved

- **InteractionChain is the audit truth.** Every secret op emits a
  Step; the Step carries fingerprint (12-hex) but never the value.
  This extends the probe_call convention from v0.11.
- **`CARL_CAMP_HF_TOKEN` invariant (2026-04-21 handoff).** When the
  managed slime dispatcher ships, it reads `CARL_CAMP_HF_TOKEN`
  through this secrets toolkit — specifically via
  `KeychainBackend.load("carl_camp_hf_token")` on carl.camp's side.
  User HF tokens stay in the user's vault, never cross the boundary.
- **NL-interpretable config (v0.16 `ConfigRequirement`).** The new
  secrets toolkit emits requirements for `HF_TOKEN`, keychain
  availability, clipboard backend, etc. Carl can answer "what do
  I need to set up secrets?" without seeing any value.
- **Tier philosophy (`tier.py:16-24`).** Gate on autonomy, not
  capability. Minting, resolving, clipboard, audit — all FREE.
  carl.camp-managed sandbox VMs for browser automation — PAID.

## Security model (what we promise)

- **In-process only.** Vault lives in the CLI process's memory.
  Never written to disk in plaintext. Persistence goes through
  `wallet_store` (Fernet-encrypted at rest, keychain-held key).
- **Revocable handles.** Every `SecretRef` can be explicitly revoked.
  Revoked refs raise `carl.secrets.revoked` on any further resolve.
- **TTL-bounded clipboard.** `ClipboardBridge.write_from_ref` schedules
  a background `threading.Timer` that wipes the clipboard after
  `ttl_s`. If the clipboard was modified externally in between
  (user copied something else), the wipe is a no-op (integrity check).
- **Audit completeness.** Every resolve emits a Step before dereferencing.
  A crash mid-operation leaves the audit trail intact; the caller can
  tell which ref was accessed and when.
- **Zero exfiltration path via Carl's context.** The agent's
  tool-call arguments carry `ref_id` (UUID) only. The agent's
  tool-call outputs carry fingerprint only. The agent's context
  never holds the raw value.

## What we do NOT promise

- Side-channel resistance against a compromised Python process. If
  an attacker can read our memory, they can read the vault.
- Hardware-attested input. Hardware-rooted trust is an admin-gated
  extension (`terminals-runtime`), not in this toolkit's scope.
- OS-level sandboxing. Running browser automation in the user's
  Playwright context means the browser has the user's permissions.
  For higher isolation, use Cua as a sandbox backend (Stage C, PAID
  tier).

## Non-goals for v0.16

- Merkle-tree proof of audit integrity (v0.17+ if carl.camp wants it).
- PKCS#11 / HSM integration (enterprise-tier, v0.18+).
- Secret sharing / Shamir split across multiple agents (research-level).
- Memory scrubbing after use (Python GC makes this lossy; we rely
  on Fernet encryption at rest for strong guarantees).

## Files to create / modify

Create:
- `packages/carl-core/src/carl_core/secrets.py` (~350 LOC)
- `packages/carl-core/tests/test_secrets.py` (~200 LOC)
- `src/carl_studio/secrets/__init__.py`
- `src/carl_studio/secrets/keychain.py` (~150 LOC)
- `src/carl_studio/secrets/clipboard.py` (~200 LOC)
- `src/carl_studio/secrets/minter.py` (~100 LOC)
- `src/carl_studio/secrets/tools.py` — tool-dispatcher registrations
- `tests/test_secrets_integration.py` (~300 LOC)
- `docs/secrets_toolkit.md`

Modify:
- `packages/carl-core/src/carl_core/interaction.py` — add four
  `ActionType` enum values
- `src/carl_studio/tool_dispatcher.py` — register the five new tools
- `packages/carl-core/src/carl_core/hashing.py` — formalize
  `fingerprint()` helper (already used inline throughout)
- `pyproject.toml` — new `[secrets]` extra, roll into `[all]`
- `docs/adapters/slime.md` — note how the secrets toolkit works
  with the forthcoming `CARL_CAMP_HF_TOKEN` managed path

Stage C additions (separate commit):
- `src/carl_studio/cu/__init__.py`
- `src/carl_studio/cu/browser.py`
- `src/carl_studio/cu/privacy.py` (forked from openadapt-privacy)
- `src/carl_studio/cu/anthropic_compat.py`
- `pyproject.toml` — new `[cu]` extra

## Verification

Unit:
- `pytest packages/carl-core/tests/test_secrets.py`
- `pytest tests/test_secrets_integration.py`

Integration (requires keyring backend + clipboard):
- `mint → copy_to_clipboard → wait 30s → verify cleared`
- `mint → hash_value → assert fingerprint stable`
- `mint → revoke → assert resolve raises carl.secrets.revoked`

Tier:
- Extend `tests/test_tier_features.py` with the 6 new feature keys.

Verify the audit trail:
- For each tool, assert exactly one Step with the expected
  `ActionType` and fingerprint lands in a fresh `InteractionChain`.
- Assert the Step's input/output NEVER contains the raw value
  (brute-force grep the serialized Step for the secret's first 4
  bytes, confirm no hit).
