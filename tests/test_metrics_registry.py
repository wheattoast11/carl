"""Tests for :mod:`carl_studio.metrics`.

Covers the lazy-import contract, singleton semantics, per-metric record
methods (including labels), the Prometheus scrape text format, and the
``is_available`` probe.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a fresh ``_LazyPrometheus`` per test.

    The module-level singleton caches a ``CollectorRegistry`` — leaking it
    across tests makes counter assertions non-deterministic because
    previous tests' increments carry forward.
    """
    from carl_studio import metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "_registry_instance", None, raising=False)


def test_registry_lazy_import_raises_clear_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing extra must surface an actionable install hint.

    We simulate absence by shadowing the import machinery so the lazy
    ``_ensure`` path takes the ``ImportError`` branch.
    """
    from carl_studio import metrics as metrics_mod

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    # Drop any cached prometheus_client modules so the lazy import path
    # actually runs through our shim.
    for mod_name in list(sys.modules):
        if mod_name == "prometheus_client" or mod_name.startswith("prometheus_client."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    if isinstance(__builtins__, dict):
        monkeypatch.setitem(__builtins__, "__import__", _fake_import)
    else:
        monkeypatch.setattr(__builtins__, "__import__", _fake_import)

    reg = metrics_mod.get_registry()
    with pytest.raises(ImportError) as exc_info:
        reg.record_training_step()
    assert "pip install 'carl-studio[metrics]'" in str(exc_info.value)


def test_registry_singleton() -> None:
    """:func:`get_registry` must return the same instance across calls."""
    from carl_studio.metrics import get_registry

    assert get_registry() is get_registry()


def test_counter_increments_training_steps() -> None:
    pytest.importorskip("prometheus_client")
    from carl_studio.metrics import get_registry

    reg = get_registry()
    reg.record_training_step()
    reg.record_training_step()
    reg.record_training_step()

    value = reg._counters["training_steps_total"]._value.get()  # type: ignore[attr-defined]
    assert value == 3.0


def test_counter_increments_phase_transitions_with_labels() -> None:
    pytest.importorskip("prometheus_client")
    from carl_studio.metrics import get_registry

    reg = get_registry()
    reg.record_phase_transition("review", "assess")
    reg.record_phase_transition("review", "assess")
    reg.record_phase_transition("assess", "plan")

    family = reg._counters["phase_transitions_total"]  # type: ignore[attr-defined]
    review_to_assess = family.labels(from_phase="review", to_phase="assess")
    assess_to_plan = family.labels(from_phase="assess", to_phase="plan")
    assert review_to_assess._value.get() == 2.0  # type: ignore[attr-defined]
    assert assess_to_plan._value.get() == 1.0  # type: ignore[attr-defined]


def test_counter_increments_x402_payments_by_status() -> None:
    pytest.importorskip("prometheus_client")
    from carl_studio.metrics import get_registry

    reg = get_registry()
    reg.record_payment("settled")
    reg.record_payment("settled")
    reg.record_payment("failed")

    family = reg._counters["x402_payments_total"]  # type: ignore[attr-defined]
    assert family.labels(status="settled")._value.get() == 2.0  # type: ignore[attr-defined]
    assert family.labels(status="failed")._value.get() == 1.0  # type: ignore[attr-defined]


def test_gauge_snapshots_queue_depths() -> None:
    pytest.importorskip("prometheus_client")
    from carl_studio.metrics import get_registry

    reg = get_registry()
    reg.snapshot_queue_depths({"queued": 5, "processing": 2, "done": 42})

    family = reg._gauges["sticky_queue_depth"]  # type: ignore[attr-defined]
    assert family.labels(status="queued")._value.get() == 5.0  # type: ignore[attr-defined]
    assert family.labels(status="processing")._value.get() == 2.0  # type: ignore[attr-defined]
    assert family.labels(status="done")._value.get() == 42.0  # type: ignore[attr-defined]

    # Overwrite semantics — gauge.set, not inc.
    reg.snapshot_queue_depths({"queued": 1})
    assert family.labels(status="queued")._value.get() == 1.0  # type: ignore[attr-defined]


def test_registry_scrape_text_format() -> None:
    """``generate_latest`` must emit the Prometheus text exposition format."""
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from carl_studio.metrics import get_registry

    reg = get_registry()
    reg.record_training_step()
    reg.record_payment("settled")
    reg.snapshot_queue_depths({"queued": 3})

    output = generate_latest(reg.registry).decode("utf-8")
    assert "# HELP carl_training_steps_total" in output
    assert "# TYPE carl_training_steps_total counter" in output
    assert "carl_training_steps_total 1.0" in output
    assert 'carl_x402_payments_total{status="settled"} 1.0' in output
    assert "# HELP carl_sticky_queue_depth" in output
    assert 'carl_sticky_queue_depth{status="queued"} 3.0' in output


def test_is_available_true_when_prometheus_installed() -> None:
    pytest.importorskip("prometheus_client")
    from carl_studio.metrics import is_available

    assert is_available() is True


# --- S2: public_registry + external-collector merge hook --------------------


def test_public_registry_returns_collector_registry() -> None:
    """:func:`public_registry` returns the shared ``CollectorRegistry``."""
    pytest.importorskip("prometheus_client")
    from prometheus_client import CollectorRegistry

    from carl_studio.metrics import public_registry

    reg = public_registry()
    assert isinstance(reg, CollectorRegistry)


def test_public_registry_identity() -> None:
    """Two calls must return the same shared :class:`CollectorRegistry`."""
    pytest.importorskip("prometheus_client")
    from carl_studio.metrics import public_registry

    assert public_registry() is public_registry()


def test_public_registry_is_wrapper_registry() -> None:
    """The public registry and the ``_LazyPrometheus.registry`` are one."""
    pytest.importorskip("prometheus_client")
    from carl_studio.metrics import get_registry, public_registry

    assert public_registry() is get_registry().registry


def test_register_external_collector_appears_in_scrape() -> None:
    """A registered external collector shows up in ``generate_latest``."""
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest
    from prometheus_client.core import GaugeMetricFamily

    from carl_studio.metrics import (
        public_registry,
        register_external_collector,
        unregister_external_collector,
    )

    class _CustomCollector:
        """Minimal collector emitting a single gauge family."""

        def collect(self) -> list[GaugeMetricFamily]:
            family = GaugeMetricFamily(
                "carl_external_test_gauge",
                "External collector test gauge.",
                value=42.0,
            )
            return [family]

    collector = _CustomCollector()
    register_external_collector(collector)
    try:
        output = generate_latest(public_registry()).decode("utf-8")
        assert "carl_external_test_gauge" in output
        assert "42.0" in output
    finally:
        unregister_external_collector(collector)


def test_unregister_external_collector_removes_from_scrape() -> None:
    """After unregister, the custom metric is no longer in the scrape."""
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest
    from prometheus_client.core import GaugeMetricFamily

    from carl_studio.metrics import (
        public_registry,
        register_external_collector,
        unregister_external_collector,
    )

    class _CustomCollector:
        def collect(self) -> list[GaugeMetricFamily]:
            return [
                GaugeMetricFamily(
                    "carl_external_unregister_gauge",
                    "Will be removed.",
                    value=7.0,
                ),
            ]

    collector = _CustomCollector()
    register_external_collector(collector)
    output_before = generate_latest(public_registry()).decode("utf-8")
    assert "carl_external_unregister_gauge" in output_before

    unregister_external_collector(collector)
    output_after = generate_latest(public_registry()).decode("utf-8")
    assert "carl_external_unregister_gauge" not in output_after


def test_public_registry_without_prometheus_installed_raises_carl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing extra surfaces ``CARLError(code="carl.metrics.unavailable")``."""
    from carl_core.errors import CARLError

    from carl_studio import metrics as metrics_mod

    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    for mod_name in list(sys.modules):
        if mod_name == "prometheus_client" or mod_name.startswith("prometheus_client."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    if isinstance(__builtins__, dict):
        monkeypatch.setitem(__builtins__, "__import__", _fake_import)
    else:
        monkeypatch.setattr(__builtins__, "__import__", _fake_import)

    with pytest.raises(CARLError) as exc_info:
        metrics_mod.public_registry()
    assert exc_info.value.code == "carl.metrics.unavailable"
    assert "carl-studio[metrics]" in str(exc_info.value)


def test_register_external_collector_without_prometheus_raises_carl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``register_external_collector`` mirrors the CARLError guard."""
    from carl_core.errors import CARLError

    from carl_studio import metrics as metrics_mod

    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    for mod_name in list(sys.modules):
        if mod_name == "prometheus_client" or mod_name.startswith("prometheus_client."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    if isinstance(__builtins__, dict):
        monkeypatch.setitem(__builtins__, "__import__", _fake_import)
    else:
        monkeypatch.setattr(__builtins__, "__import__", _fake_import)

    class _Dummy:
        def collect(self) -> list[object]:
            return []

    with pytest.raises(CARLError) as exc_info:
        metrics_mod.register_external_collector(_Dummy())  # type: ignore[arg-type]
    assert exc_info.value.code == "carl.metrics.unavailable"
