"""Tests for carl_studio.adapters — TrainingConnection integration.

Verifies:
  * every built-in adapter now inherits TrainingConnection (and therefore
    BaseConnection) while still satisfying the duck-typed UnifiedBackend
    protocol;
  * the Connection lifecycle round-trips cleanly for a TRL adapter;
  * InteractionChain records the expected event names when an adapter is
    driven through ``open`` / ``submit_with_telemetry`` / ``close``;
  * a TrainingConnection subclass that omits a backend-specific spec still
    inherits the module default and boots correctly.
"""
from __future__ import annotations

from typing import Any

import pytest

from carl_core.connection import BaseConnection, ConnectionState
from carl_core.interaction import InteractionChain
from carl_studio.adapters import (
    AtroposAdapter,
    AxolotlAdapter,
    TinkerAdapter,
    TRLAdapter,
    TrainingConnection,
    UnifiedBackend,
    UnslothAdapter,
)
from carl_studio.adapters.connection import DEFAULT_TRAINING_SPEC
from carl_studio.adapters.protocol import BackendJob, BackendStatus


_ALL_BUILTINS: tuple[type[TrainingConnection], ...] = (
    TRLAdapter,
    UnslothAdapter,
    AxolotlAdapter,
    AtroposAdapter,
    TinkerAdapter,
)


@pytest.mark.parametrize("cls", _ALL_BUILTINS)
def test_builtin_inherits_training_connection(cls: type[TrainingConnection]) -> None:
    assert issubclass(cls, TrainingConnection)
    assert issubclass(cls, BaseConnection)


@pytest.mark.parametrize("cls", _ALL_BUILTINS)
def test_builtin_satisfies_unified_backend_protocol(
    cls: type[TrainingConnection],
) -> None:
    instance = cls()
    assert isinstance(instance, UnifiedBackend)


@pytest.mark.parametrize("cls", _ALL_BUILTINS)
def test_builtin_has_backend_specific_spec(cls: type[TrainingConnection]) -> None:
    instance = cls()
    assert instance.spec is not DEFAULT_TRAINING_SPEC
    assert instance.spec.name.startswith("carl.training.")


def test_trl_adapter_lifecycle_transitions_to_ready_and_closed() -> None:
    adapter = TRLAdapter()
    assert adapter.state == ConnectionState.INIT
    adapter.open()
    assert adapter.state == ConnectionState.READY
    adapter.close()
    assert adapter.state == ConnectionState.CLOSED


def test_trl_adapter_context_manager_round_trip() -> None:
    with TRLAdapter() as adapter:
        assert adapter.state == ConnectionState.READY
        job = adapter.submit({"base_model": "gpt2", "dataset_repo": "foo/bar"})
        assert isinstance(job, BackendJob)
        assert job.backend == "trl"
        assert job.status == BackendStatus.PENDING
    assert adapter.state == ConnectionState.CLOSED


def test_trl_submit_with_telemetry_records_events() -> None:
    chain = InteractionChain()
    adapter = TRLAdapter(chain=chain)
    adapter.open()
    job = adapter.submit_with_telemetry(
        {"base_model": "gpt2", "dataset_repo": "foo/bar"}
    )
    adapter.close()

    names = [step.name for step in chain.steps]
    assert "connection.open" in names
    assert "connection.submit" in names
    assert "connection.submit.job" in names
    assert "connection.close" in names
    assert job.backend == "trl"


def test_subclass_without_override_falls_back_to_default_spec() -> None:
    class MinimalBackend(TrainingConnection):
        name = "minimal"

        def available(self) -> bool:  # type: ignore[override]
            return True

        def submit(self, carl_config: dict[str, Any]) -> BackendJob:  # type: ignore[override]
            return BackendJob(run_id="min-1", backend=self.name)

        def status(self, run_id: str) -> BackendJob:  # type: ignore[override]
            return BackendJob(run_id=run_id, backend=self.name)

        def logs(self, run_id: str, *, tail: int = 100) -> list[str]:  # type: ignore[override]
            return []

        def cancel(self, run_id: str) -> bool:  # type: ignore[override]
            return False

    adapter = MinimalBackend()
    assert adapter.spec is DEFAULT_TRAINING_SPEC
    adapter.open()
    with adapter.transact("submit"):
        job = adapter.submit({})
    adapter.close()
    assert job.run_id == "min-1"


def test_connect_raises_if_backend_unavailable() -> None:
    class UnreachableBackend(TrainingConnection):
        spec = DEFAULT_TRAINING_SPEC
        name = "unreachable"

        def available(self) -> bool:  # type: ignore[override]
            return False

        def submit(self, carl_config: dict[str, Any]) -> BackendJob:  # type: ignore[override]
            raise AssertionError("unreachable")

        def status(self, run_id: str) -> BackendJob:  # type: ignore[override]
            raise AssertionError

        def logs(self, run_id: str, *, tail: int = 100) -> list[str]:  # type: ignore[override]
            raise AssertionError

        def cancel(self, run_id: str) -> bool:  # type: ignore[override]
            raise AssertionError

    adapter = UnreachableBackend()
    with pytest.raises(Exception):
        adapter.open()
    assert adapter.state == ConnectionState.ERROR
    adapter.close()
    assert adapter.state == ConnectionState.CLOSED


def test_duck_typed_consumer_sees_connection_and_legacy_both() -> None:
    """A function that just reads .name / .available() works unchanged, and
    a function that wants the full Connection API can use it."""
    adapter = TRLAdapter()
    # duck-typed path
    assert adapter.name == "trl"
    assert adapter.available() is True
    # new connection path
    assert adapter.connection_id.startswith("conn-")
    assert adapter.spec.name == "carl.training.trl"
