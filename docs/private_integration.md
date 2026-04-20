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
