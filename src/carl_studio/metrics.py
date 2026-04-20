"""Prometheus metrics registry for CARL Studio.

Singleton pattern with lazy import so ``import carl_studio.metrics`` never
fails — only actual instrumentation methods raise ``ImportError`` when the
``[metrics]`` extra is missing. Record methods are cheap no-ops when the
extra is absent (``record_*`` guards at the call site should still gate
with :func:`is_available`, but double-gating is safe).

All counters/gauges are registered on a shared :class:`CollectorRegistry`
owned by the process singleton. The registry is deliberately not the
``prometheus_client`` default so tests stay isolated and multiple processes
(CLI + daemon) can coexist without label collisions.

Metric naming follows Prometheus convention: ``carl_<subsystem>_<name>``
with the ``_total`` suffix for counters per the
``https://prometheus.io/docs/practices/naming/`` style guide.

Public API
----------

- :func:`get_registry` — internal ``_LazyPrometheus`` wrapper with
  ``record_*`` helpers. Stable but CARL-internal.
- :func:`public_registry` — the shared :class:`CollectorRegistry` itself.
  Private dashboards and deployment infrastructure can register additional
  collectors via :func:`register_external_collector` or attach scrapers
  that read from this registry directly.
- :func:`register_external_collector` /
  :func:`unregister_external_collector` — mount or remove external
  :class:`prometheus_client.registry.Collector` subclasses on the shared
  registry without monkey-patching. Inverse pair is suitable for test
  teardown.
- :func:`is_available` — probe for the optional ``[metrics]`` extra.

When ``prometheus_client`` is not installed, :func:`public_registry` and
:func:`register_external_collector` raise
:class:`carl_core.errors.CARLError` with code ``carl.metrics.unavailable``
and a hint to install the ``metrics`` extra.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any

from carl_core.errors import CARLError

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry
    from prometheus_client.registry import Collector


class _LazyPrometheus:
    """Lazy wrapper around ``prometheus_client``.

    Raises a clear install hint when the metrics extra is missing, but
    module import never fails. The registry and all metric objects are
    constructed on first touch under :attr:`_lock` so concurrent callers
    never double-register.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._registry: CollectorRegistry | None = None
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}

    def _ensure(self) -> None:
        """Materialize the registry and metric families. Idempotent."""
        if self._registry is not None:
            return
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge
        except ImportError as e:
            raise ImportError(
                "prometheus_client not installed. Install the metrics extra: "
                "pip install 'carl-studio[metrics]'"
            ) from e
        with self._lock:
            if self._registry is not None:
                return
            registry = CollectorRegistry()
            # Counters
            self._counters["training_steps_total"] = Counter(
                "carl_training_steps_total",
                "Training steps completed.",
                registry=registry,
            )
            self._counters["phase_transitions_total"] = Counter(
                "carl_phase_transitions_total",
                "Phase transitions.",
                ["from_phase", "to_phase"],
                registry=registry,
            )
            self._counters["crystallizations_total"] = Counter(
                "carl_crystallizations_total",
                "Crystallizations.",
                registry=registry,
            )
            self._counters["heartbeat_cycles_total"] = Counter(
                "carl_heartbeat_cycles_total",
                "Heartbeat cycles completed.",
                registry=registry,
            )
            self._counters["heartbeat_maintenance_failures_total"] = Counter(
                "carl_heartbeat_maintenance_failures_total",
                "Maintenance failures.",
                registry=registry,
            )
            self._counters["x402_payments_total"] = Counter(
                "carl_x402_payments_total",
                "x402 payments.",
                ["status"],
                registry=registry,
            )
            self._counters["rate_limit_hits_total"] = Counter(
                "carl_rate_limit_hits_total",
                "Anthropic rate-limit hits.",
                registry=registry,
            )
            # Gauges
            self._gauges["sticky_queue_depth"] = Gauge(
                "carl_sticky_queue_depth",
                "Sticky queue depth by status.",
                ["status"],
                registry=registry,
            )
            # Assign last so a partially-built state is never observable.
            self._registry = registry

    @property
    def registry(self) -> CollectorRegistry:
        """The underlying :class:`CollectorRegistry`. Triggers lazy init."""
        self._ensure()
        assert self._registry is not None
        return self._registry

    # -- counter recorders ------------------------------------------------

    def record_training_step(self) -> None:
        """Increment ``carl_training_steps_total``."""
        self._ensure()
        self._counters["training_steps_total"].inc()

    def record_phase_transition(self, from_phase: str, to_phase: str) -> None:
        """Increment ``carl_phase_transitions_total`` by label pair."""
        self._ensure()
        self._counters["phase_transitions_total"].labels(
            from_phase=from_phase,
            to_phase=to_phase,
        ).inc()

    def record_crystallization(self) -> None:
        """Increment ``carl_crystallizations_total``."""
        self._ensure()
        self._counters["crystallizations_total"].inc()

    def record_heartbeat_cycle(self) -> None:
        """Increment ``carl_heartbeat_cycles_total``."""
        self._ensure()
        self._counters["heartbeat_cycles_total"].inc()

    def record_maintenance_failure(self) -> None:
        """Increment ``carl_heartbeat_maintenance_failures_total``."""
        self._ensure()
        self._counters["heartbeat_maintenance_failures_total"].inc()

    def record_payment(self, status: str) -> None:
        """Increment ``carl_x402_payments_total`` with a status label."""
        self._ensure()
        self._counters["x402_payments_total"].labels(status=status).inc()

    def record_rate_limit_hit(self) -> None:
        """Increment ``carl_rate_limit_hits_total``."""
        self._ensure()
        self._counters["rate_limit_hits_total"].inc()

    # -- gauge recorders --------------------------------------------------

    def snapshot_queue_depths(self, counts: dict[str, int]) -> None:
        """Set ``carl_sticky_queue_depth{status=...}`` for each entry."""
        self._ensure()
        gauge = self._gauges["sticky_queue_depth"]
        for status, value in counts.items():
            gauge.labels(status=status).set(value)


_registry_instance: _LazyPrometheus | None = None
_registry_instance_LOCK = Lock()


def get_registry() -> _LazyPrometheus:
    """Return the process-wide singleton :class:`_LazyPrometheus`.

    Double-checked locking keeps the hot path allocation-free once the
    singleton is materialized.
    """
    global _registry_instance
    if _registry_instance is None:
        with _registry_instance_LOCK:
            if _registry_instance is None:
                _registry_instance = _LazyPrometheus()
    return _registry_instance


def is_available() -> bool:
    """``True`` if ``prometheus_client`` can be imported.

    Callers should gate metrics hooks with this in hot paths so a missing
    extra degrades gracefully instead of raising.
    """
    try:
        import importlib

        importlib.import_module("prometheus_client")
        return True
    except ImportError:
        return False


_METRICS_UNAVAILABLE_MSG = (
    "prometheus_client not installed. Install the metrics extra: "
    "pip install 'carl-studio[metrics]'"
)


def public_registry() -> CollectorRegistry:
    """Return the shared Prometheus :class:`CollectorRegistry`.

    Private dashboards and deployment infrastructure can register
    additional collectors via :func:`register_external_collector` or attach
    scrapers that read from this registry directly. Calls
    :meth:`_LazyPrometheus._ensure` to materialize the registry on first
    touch; subsequent calls return the same instance (singleton).

    Raises
    ------
    CARLError
        With ``code="carl.metrics.unavailable"`` when the ``[metrics]``
        extra is not installed.
    """
    wrapper = get_registry()
    try:
        return wrapper.registry
    except ImportError as exc:
        raise CARLError(
            _METRICS_UNAVAILABLE_MSG,
            code="carl.metrics.unavailable",
            cause=exc,
        ) from exc


def register_external_collector(collector: Collector) -> None:
    """Register an external :class:`prometheus_client.registry.Collector`.

    The collector is attached to the shared registry returned by
    :func:`public_registry`. Values produced by ``collector.collect()``
    appear in every subsequent scrape of that registry — including the
    endpoint served by ``carl metrics serve``.

    Raises
    ------
    CARLError
        With ``code="carl.metrics.unavailable"`` when the ``[metrics]``
        extra is not installed.
    """
    registry = public_registry()
    registry.register(collector)


def unregister_external_collector(collector: Collector) -> None:
    """Inverse of :func:`register_external_collector`.

    Intended for test teardown and dynamic reconfiguration flows.

    Raises
    ------
    CARLError
        With ``code="carl.metrics.unavailable"`` when the ``[metrics]``
        extra is not installed.
    """
    registry = public_registry()
    registry.unregister(collector)


__all__ = [
    "get_registry",
    "is_available",
    "public_registry",
    "register_external_collector",
    "unregister_external_collector",
]
