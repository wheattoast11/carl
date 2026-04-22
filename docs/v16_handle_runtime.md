---
last_updated: 2026-04-21
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.16.1
---

# The CARL handle runtime

> Carl orchestrates value transfer and environment interaction by moving
> **references**, not values. Every operation that crosses a tool boundary
> emits an audit step carrying a fingerprint — never the bytes.

This is the unifying doctrine behind four toolkits shipped in the v0.16
line: **secrets**, **data**, **resource**, and **browser/computer-use**.
They are not four separate systems. They are four specializations of one
capability-constrained handle runtime.

## Mental model

Carl is an AI-native intelligence. It reasons in English. Your job is to
give it a grammar where values it shouldn't see, values it should see but
shouldn't leak to external systems, and long-lived resources it needs to
address across multiple steps all look like **the same thing**: an opaque
handle with a fingerprint, a kind, and a describe method.

Four layers, one shape:

| Layer       | `Ref` type      | `Vault` type      | `Toolkit` type                  | Install extra |
|-------------|-----------------|-------------------|---------------------------------|---------------|
| Secrets     | `SecretRef`     | `SecretVault`     | `SecretsToolkit`                | `[secrets]`   |
| Data        | `DataRef`       | `DataVault`       | `DataToolkit`                   | (built-in)    |
| Resource    | `ResourceRef`   | `ResourceVault`   | `BrowserToolkit`, ...           | `[cu]` (soon) |
| Computer-use| — (reuses above)| — (reuses above)  | `CUDispatcher` (Anthropic compat)| `[cu]` (soon) |

Every `Ref`:

- is **frozen** (Pydantic `frozen=True, extra="forbid"`),
- carries a `ref_id: UUID`, a `kind: Literal[...]`, a `uri: str`, and a `ttl_s: int | None`,
- has `.describe() → dict` returning metadata-only — no raw value,
- has `.is_expired()` semantics checked lazily at resolve time.

Every `Vault`:

- is thread-safe (RLock),
- exposes `put* / resolve(privileged=True) / revoke / list_refs / exists`,
- never serializes the backend — only the ref descriptor,
- requires explicit `privileged=True` to dereference (for `Secret` and
  `Resource`; `Data` reads are non-privileged but cap preview bytes).

Every `Toolkit`:

- is a dataclass wrapping `(vault, chain, ...)`,
- exposes agent-callable methods returning JSON-native dicts,
- emits one audit step per method call (action types listed below),
- exposes `tool_schemas() → list[dict]` for Anthropic-tool-style registration.

## Audit contract

Every op lands in the active `InteractionChain` as a `Step` with an
action type from this list (all introduced in v0.16):

| Action                | Emitter                                  | What it says without what it says |
|-----------------------|------------------------------------------|------------------------------------|
| `SECRET_MINT`         | `CryptoRandomMinter`                     | new credential shape + 12-hex fingerprint |
| `SECRET_RESOLVE`      | `KeychainBackend` / `SecretsToolkit`     | privileged deref happened          |
| `SECRET_REVOKE`       | `SecretVault.revoke`                     | handle invalidated                 |
| `CLIPBOARD_WRITE`     | `ClipboardBridge`                        | fingerprint + TTL, not value       |
| `DATA_OPEN`           | `DataToolkit.open_*`                     | kind + uri + size + sha256 prefix  |
| `DATA_READ`           | `DataToolkit.read/read_text/read_json`   | offset + length_returned + sha12   |
| `DATA_TRANSFORM`      | `DataToolkit.transform`                  | op + source_fingerprint + derived descriptor |
| `DATA_PUBLISH`        | `DataToolkit.publish_to_file`            | destination + bytes_written + sha12 |
| `RESOURCE_OPEN`       | `BrowserToolkit.open_page`               | page descriptor — provider / uri   |
| `RESOURCE_ACT`        | `BrowserToolkit.*` / `CUDispatcher.dispatch` | action name + ref_id + structural args |
| `RESOURCE_CLOSE`      | `BrowserToolkit.close_page`              | closed=True/False                  |

Chain serialization (`to_jsonl()` → `~/.carl/interactions/*.jsonl`) also
runs the secret-shape scrubbers, so even if a toolkit bug tried to land a
raw secret in a Step, it still gets scrubbed at persistence time. Defence
in depth.

## The capability-security core

The model is **object-capability** (Mark Miller, E-rights). The reference
to a value IS the authority to operate on it. You can:

- **Pass a ref** → grants: operate on the value via the toolkit.
- **Withhold a ref** → denies: no way to address the value.
- **Revoke a ref** → invalidates: future resolves raise `*Error` codes.

Compared to role/permission systems:

- No global "secrets are admin-only" flag. Every call site that wants the
  value passes the ref into `vault.resolve(ref, privileged=True)`. The
  `privileged=True` isn't enforcement — it's *visibility*. It forces
  every value-access to appear in a diff, so code review can spot them.
- Handles are the grain of permission. Grant a browser page ref to the
  "navigation" subroutine; grant the same ref + a secret ref to the
  "login" subroutine. No global kitchen sink.
- Revocation is **synchronous and local**: you don't rotate a key and
  hope the token server expires the old one. You flip a bit in the vault
  and the next `resolve()` raises.

## Why four toolkits instead of one

Because the *grammar* is the same but the *risk profile* differs:

**Secrets.** Default is zero-knowledge — the agent never sees the value.
`SecretVault.resolve` demands `privileged=True` and raises a `SecretsError`
code `carl.secrets.unauthorized_resolve` otherwise. Use for: API keys,
passwords, recovery phrases, PII.

**Data.** The agent often *needs* to see the value (it's reading a JSON
response, a file, a query result). But reads are bounded: `read()` caps
at `preview_bytes` (default 64 KB) unless explicit `length` is passed,
and `length > max_read_bytes` is refused. Reads return base64'd bytes in
a structured dict so large payloads don't accidentally flow as strings
through agent context.

**Resource.** Long-lived, stateful, closeable. Carl needs to reference
"this browser page" across many steps. The backend (a `Page`, a
`Popen`, an MCP session) lives behind the vault and dispatches via
toolkit methods. `revoke()` runs a caller-supplied `closer(backend)` so
close lifecycles stay local to the toolkit that knows how.

**Computer-use.** Not a new vault layer — a re-shape of the browser
toolkit's surface to match Anthropic's `computer_20250124` tool schema.
`CUDispatcher.bind_page(ref_id)` + `dispatch({"action": "left_click",
"coordinate": [x, y]})` maps cleanly to `BrowserToolkit.page_from_id`
and the page's mouse methods. Screenshots return a `DataRef` descriptor;
Carl sees the handle, the PNG bytes live in the data vault.

## End-to-end example: Carl logs into a site without ever seeing the password

```python
from carl_core.interaction import InteractionChain
from carl_core.resource_handles import ResourceVault
from carl_core.secrets import SecretVault
from carl_studio.handles.data import DataToolkit
from carl_studio.cu.browser import BrowserToolkit
from carl_studio.cu.anthropic_compat import CUDispatcher
from carl_studio.secrets.keychain import KeychainBackend
from carl_studio.secrets.minter import CryptoRandomMinter

chain = InteractionChain()
secrets = SecretVault()
data = DataToolkit.build(chain)
resources = ResourceVault()
browser = BrowserToolkit.build(
    chain, data_toolkit=data, secret_vault=secrets, resource_vault=resources
)

# (1) Load password from keychain into the vault (returns a SecretRef).
keychain = KeychainBackend(secrets, chain=chain)
pw_ref = keychain.load_to_vault(service="gmail", account="tej@x.tech")

# (2) Carl navigates.
page = browser.open_page(url="https://mail.example.com/login")
browser.type_text(page["ref_id"], "#email", "tej@x.tech")

# (3) Type the password — value resolved INSIDE the toolkit,
#     never crosses a Carl tool-call boundary.
browser.type_from_secret(page["ref_id"], "#password", str(pw_ref.ref_id))
browser.click(page["ref_id"], "#login")

# (4) Capture post-login screenshot into the data vault.
shot = browser.screenshot(page["ref_id"])
# Carl sees: shot["data_ref"] = {"ref_id": ..., "size_bytes": 34218, ...}
# Carl does NOT see: the PNG bytes.

# (5) Serialize the chain — no secret value, no image bytes leak.
jsonl = chain.to_jsonl()
assert "hunter2" not in jsonl  # (if that were the password)
```

Carl's mental model is:

- "I have `pw_ref`, which is the handle to the password."
- "I told the browser to type `pw_ref` into `#password`."
- "I got back `shot.data_ref`, which is the handle to the screenshot."

It never wrote the password. It never saw the PNG bytes. Every step is
in the chain as a 12-hex fingerprint trail.

## Wiring into CARLAgent

Each toolkit exposes `tool_schemas() → list[dict]` in the shape CARLAgent's
tool dispatcher already understands:

```python
from carl_studio.chat_agent import CARLAgent

agent = CARLAgent(...)

# Register the data toolkit's tools
for schema in data.tool_schemas():
    agent.register_tool(
        name=schema["name"],
        description=schema["description"],
        input_schema=schema["input_schema"],
        handler=getattr(data, schema["name"].removeprefix("data_")),
    )

# Register the browser toolkit's tools the same way
for schema in browser.tool_schemas():
    ...

# Register the Anthropic computer-use tool as a single entry
from carl_studio.cu.anthropic_compat import COMPUTER_USE_TOOL_SCHEMA, CUDispatcher

cu = CUDispatcher(browser=browser)
agent.register_tool(
    name="computer",
    description=COMPUTER_USE_TOOL_SCHEMA["description"],
    input_schema=COMPUTER_USE_TOOL_SCHEMA["input_schema"],
    handler=cu.dispatch,
)
```

The agent prompt should explain the grammar once: *"You receive handles
called ref_ids. To act on a value, call the toolkit method that accepts
the ref_id. You never need to see the raw bytes — ask the toolkit for a
transformation (`read_text`, `transform`, etc.) and it will emit a new
handle."*

## Tier-gating

All handle-runtime features are **FREE**. See `packages/carl-core/src/carl_core/tier.py`:

```python
"data.open": Tier.FREE,
"data.read": Tier.FREE,
"data.transform": Tier.FREE,
"data.publish": Tier.FREE,
"resource.open": Tier.FREE,
"resource.act": Tier.FREE,
"resource.close": Tier.FREE,
```

Rationale: the handle runtime IS how Carl reasons about values it
shouldn't see. Gating it would break Carl as a viable agent, not just
pay-wall convenience. Autonomy features that *coordinate* handles across
long-running workflows (scheduled agent runs, managed-fleet orchestration)
stay PAID. Capability-constrained value transfer is table stakes.

## What lives where

- **Primitives (carl-core, zero network, minimal deps):**
  - `carl_core/secrets.py` — `SecretRef`, `SecretVault`, seal/unseal
  - `carl_core/data_handles.py` — `DataRef`, `DataVault`
  - `carl_core/resource_handles.py` — `ResourceRef`, `ResourceVault`
  - `carl_core/interaction.py` — `ActionType` (SECRET_*, DATA_*, RESOURCE_*)

- **Integrations (carl-studio, optional deps lazy-imported):**
  - `carl_studio/secrets/` — minter, keychain, clipboard, `SecretsToolkit`
  - `carl_studio/handles/data.py` — `DataToolkit`
  - `carl_studio/cu/browser.py` — `BrowserToolkit` (Playwright)
  - `carl_studio/cu/anthropic_compat.py` — `CUDispatcher`, `COMPUTER_USE_TOOL_SCHEMA`
  - `carl_studio/cu/privacy.py` — content-level PII redaction

## One-call wiring via `HandleRuntimeBundle`

Manually building every vault + toolkit + registering every handler is
~40 lines of boilerplate future Claude sessions will get slightly wrong.
The bundle reduces it to four:

```python
from carl_core.interaction import InteractionChain
from carl_studio.handles import HandleRuntimeBundle

chain = InteractionChain()
bundle = HandleRuntimeBundle.build(chain)
bundle.register_all(agent.tool_dispatcher)           # registers 25 tools
tools_for_anthropic = bundle.anthropic_tools()       # flat schema list
# ... pass `tools=tools_for_anthropic` to the Anthropic API call
```

`register_all()` accepts any object with a `.register(name, fn)` method
(duck-typed against `ToolDispatcher`). The `make_handler()` shim wraps
each toolkit method in the `(dict → (str, bool))` ToolCallable contract:
JSON-encoded result on success, `("Error: ...", True)` on exception.

Full surface registered:

- **Data** (6): `data_open_file`, `data_read_text`, `data_read_json`,
  `data_transform`, `data_publish_to_file`, `data_list_handles`.
- **Browser** (11): `browser_open_page`, `browser_navigate`,
  `browser_click`, `browser_type_text`, `browser_type_from_secret`,
  `browser_press_key`, `browser_scroll`, `browser_screenshot`,
  `browser_extract_text`, `browser_close_page`, `browser_list_pages`.
- **Subprocess** (7): `subprocess_spawn`, `subprocess_poll`,
  `subprocess_wait`, `subprocess_terminate`, `subprocess_read_stdout`,
  `subprocess_read_stderr`, `subprocess_list`.
- **Computer-use** (1): `computer` (single Anthropic-shape tool;
  dispatcher fans out to browser mouse/keyboard/screenshot).

`bundle.tool_catalog()` returns a Carl-readable description of the
full surface — handy for a "what can you do?" meta-tool the agent
can introspect at runtime.

## SubprocessToolkit quick reference

```python
tk = bundle.subprocess_toolkit

# Spawn. argv-only — no shell. Default TTL 300s.
proc = tk.spawn(["python", "train.py", "--config", "carl.yaml"])

# Monitor.
status = tk.poll(proc["ref_id"])             # running? exit_code?
chunk = tk.read_stdout(proc["ref_id"])       # buffered stdout → DataRef

# Drain + collect.
result = tk.wait(proc["ref_id"], timeout_s=600)
#   result["exit_code"] — int
#   result["stdout_ref"] / ["stderr_ref"] — DataRef descriptors

# Kill + cleanup.
tk.terminate(proc["ref_id"], grace_s=5.0)    # SIGTERM then SIGKILL
```

The shell-string rejection is a type-level contract: `argv: list[str]`.
This is the capability-security model applied to shell injection —
the dangerous value (a shell string) simply can't be expressed via
the toolkit API. If you need pipe composition or globbing, either
build the argv (`["sh", "-c", "foo | bar"]` is allowed but explicit)
or chain multiple spawns.

## Known gaps (deferred)

- `ResourceVault` adapters for MCP sessions / rollout engines. Shape
  is ready; toolkit layer pending.
- `BrowserToolkit` is Playwright sync-api only. Async-capable agents
  should wrap via `anyio.from_thread` for now; a native async toolkit
  can land later sharing the same vault.
- `DataToolkit.transform` supports head / tail / gzip / gunzip / digest.
  Extend with `csv`, `json_path`, `html_parse` as Carl's needs surface.
- `privacy.redact_text` is regex-only. Swap in openadapt's ML-assisted
  redactor when the v0.16 secrets-toolkit consumer asks for it.

## Sources

- Mark S. Miller, *Robust Composition* (2006) — the object-capability model.
- HashiCorp Vault — response wrapping primitive (single-use tokens).
- 1Password `op://` URI shape — inspiration for the `carl://` ref format.
- Anthropic Computer Use docs — `computer_20250124` tool schema.
- `docs/v16_secrets_toolkit_design.md` — Stage A/B design notes.
- `docs/v16_utils_inventory.md` — library picks that back these layers.
