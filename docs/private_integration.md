---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.8.0
---

# Private-runtime integration guide

carl-studio is MIT-licensed; `terminals-runtime`, `carl-admin`, and the
carl.camp backend are separate repos with their own licenses. v0.8.0
adds three named plug-points that let those private surfaces extend
carl-studio's behavior without monkey-patching.

This doc shows the minimal working example for each.

## 1. x402 payment-confirmation callback

**Problem.** `X402Config.confirm_payment_cb` can hold either a `Callable`
(passed directly at construction) or a string name (persisted via
carl.camp account settings). v0.7.1 only supported the Callable path —
string names had no resolution mechanism.

**Plug-point (v0.8.0).**

```python
from carl_studio.x402 import (
    register_confirm_callback,
    unregister_confirm_callback,
    X402Client,
    X402Config,
)

def always_approve_under_one_dollar(**ctx) -> bool:
    return float(ctx.get("amount", 0.0)) < 1.0

register_confirm_callback("approve_small", always_approve_under_one_dollar)

config = X402Config(
    facilitator_url="https://facilitator.example.com",
    confirm_payment_cb="approve_small",   # string — resolved at execute
    daily_spend_cap=10.0,
)
client = X402Client(config=config, db=localdb)
```

Direct `Callable` passing still works; the registry is only consulted
when `confirm_payment_cb` is a `str`. Unregistered names raise
`CARLError(code="carl.x402.callback_unregistered")` at `.execute()`.

## 2. Metrics external-collector merge

**Problem.** v0.7.1 exposed `carl metrics serve` but the underlying
`CollectorRegistry` was private. Private dashboards couldn't attach
custom collectors without monkey-patching `metrics._registry`.

**Plug-point (v0.8.0).**

```python
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from carl_studio.metrics import public_registry, register_external_collector

class PrivateQueueDepthCollector(Collector):
    def collect(self):
        g = GaugeMetricFamily(
            "carl_private_queue_depth",
            "Depth of the internal private-runtime queue.",
        )
        g.add_metric([], value=fetch_private_queue_depth())
        yield g

register_external_collector(PrivateQueueDepthCollector())

# The collector now surfaces on carl metrics serve (same registry).
# Direct registry access is also available:
registry = public_registry()
```

If the `metrics` extra isn't installed, both calls raise
`CARLError(code="carl.metrics.unavailable")` with an install hint.

## 3. Pluggable tier resolver

**Problem.** `TierPredicate._effective()` hardcoded `detect_effective_tier()`
via `CARLSettings`. Private runtimes with alternate tier-source semantics
(wallet-balance, JWT-claim, per-feature overrides) had to monkey-patch.

**Plug-point (v0.8.0).**

```python
from carl_studio.tier import Tier, register_tier_resolver, clear_tier_resolver

def jwt_claim_resolver(feature: str | None) -> Tier:
    claims = current_request_claims()       # provided by your JWT middleware
    if claims.get("tier") == "paid":
        return Tier.PAID
    if feature and feature in claims.get("feature_overrides", []):
        return Tier.PAID
    return Tier.FREE

register_tier_resolver(jwt_claim_resolver)

# All subsequent @tier_gate(Tier.PAID, feature="...") calls consult the
# resolver before falling back to detect_effective_tier(). The resolver
# receives the feature name for per-feature decisions.

# Test teardown:
clear_tier_resolver()
```

Resolver exceptions wrap as `CARLError(code="carl.tier.resolver_error",
context={"inner": str(exc)})` with the original `__cause__` preserved.
Registration replaces any prior resolver — only one is active at a time.

## Error codes at a glance

| Code | Raised by |
|------|-----------|
| `carl.x402.callback_unregistered` | `X402Client.execute` when `confirm_payment_cb` is a string with no registered callback |
| `carl.metrics.unavailable` | `public_registry()` / `register_external_collector()` when prometheus-client isn't installed |
| `carl.tier.resolver_error` | `TierPredicate._effective()` when a registered resolver raises |
| `carl.config.schema_mismatch` | `ConfigRegistry.get()` when the stored JSON fails Pydantic validation |
| `carl.resilience.circuit_open` | `BreakAndRetryStrategy.run()` when the breaker is open |

All follow the `carl.<namespace>.<specific>` convention and inherit
from `carl_core.errors.CARLError` — `.to_dict()` auto-redacts
secret-shaped keys.

## `load_private()` — the canonical BUSL↔MIT seam (v0.8+)

`src/carl_studio/admin.py::load_private(module_name)` is the authoritative
pattern for lazy-importing terminals-runtime primitives into MIT
carl-studio code.

### Fallback contract

Every consumer of `load_private()` MUST preserve behavior when the
private module is unavailable. The pattern:

```python
# packages/carl-core/src/carl_core/coherence_observer.py  (reference)
def _load_prompt() -> str:
    try:
        from terminals_runtime.observe import OBSERVER_SYSTEM_PROMPT  # type: ignore
        return OBSERVER_SYSTEM_PROMPT
    except ImportError:
        # MIT-safe fallback: minimal prompt that preserves contract,
        # not the proprietary watchpoint methodology.
        return _MINIMAL_FALLBACK_PROMPT
```

### Three layers of the gate

`load_private()` consults three gates in order:

1. **Hardware-HMAC** — `hmac(IOPlatformSerialNumber + processor +
   hostname, CARL_ADMIN_SECRET)` matches the stored key in
   `~/.carl/admin.key`. Fails fast if any of the three inputs don't
   match; no retry.
2. **terminals-runtime package present** — `pip show
   terminals-runtime` succeeds. When it does, direct import wins
   (no HF download needed).
3. **HF private dataset** — falls back to downloading the matching
   module from `wheattoast11/carl-private` via `huggingface_hub`.

If all three fail, `ImportError` is raised with a user-facing message
(`"Install terminals-runtime or contact tej@terminals.tech for admin
access"`). The caller's job is to catch this and degrade to the
MIT-safe fallback — **never** to propagate the ImportError to the end
user as if the feature is broken.

### Non-obligations

- Callers are NOT required to check `admin.is_admin()` before trying
  `load_private()`. The gate is internal to the loader.
- Callers are NOT required to cache the loaded module; consult the
  private-runtime seam each call. The loader itself caches sys.modules
  after first import.
- Callers MUST preserve public contract under the fallback path.
  Never expose "this feature is private" as a user-visible error.

## Testing isolation

Each plug-point uses an in-process registry. Tests that mutate any of
them should add teardown:

```python
@pytest.fixture(autouse=True)
def _cleanup_registries():
    yield
    from carl_studio.x402 import _CALLBACK_REGISTRY
    from carl_studio.tier import clear_tier_resolver
    _CALLBACK_REGISTRY.clear()
    clear_tier_resolver()
    # Metrics external collectors — track and unregister explicitly
    # in each test; the registry is module-scoped.
```
